"""Arquitetura e wrapper sklearn-compatible da rede neural para churn."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class FocalLoss(nn.Module):
    """Focal Loss para classificação binária com dados desbalanceados.

    Reduz a contribuição de exemplos fáceis (bem classificados) e concentra
    o aprendizado nos exemplos difíceis — mais eficaz que pos_weight sozinho.

    alpha: peso geral da loss
    gamma: fator de foco (0 = BCE puro; 2 = padrão recomendado)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


class ChurnNet(nn.Module):
    """MLP para classificação binária de churn: 128 → 64 → 32 → 1."""

    def __init__(self, input_dim: int, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = [0.3, 0.2]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout[0]),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout[1]),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class ChurnNetWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper sklearn-compatible para ChurnNet (PyTorch).

    Definido em utils/neural_net.py para ser importável pelos workers
    do joblib (RandomizedSearchCV com n_jobs=-1 usa multiprocessing,
    que serializa o Pipeline via pickle — classes definidas em células
    de notebook não são resolvíveis nos workers).

    Melhorias em relação à versão original:
    - Focal Loss (use_focal_loss=True): foca nos exemplos difíceis
    - Cosine Annealing com warmup: melhor convergência que ReduceLROnPlateau
    - Gradient clipping: estabiliza o treino
    - Early stopping por ROC-AUC de validação (val_fraction)
    """

    def __init__(self, lr=1e-3, weight_decay=1e-3, dropout=None,
                 epochs=500, batch_size=64, patience=50, use_smote=True,
                 use_focal_loss=True, focal_gamma=2.0, warmup_epochs=10,
                 val_fraction=0.1):
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.dropout       = dropout if dropout is not None else [0.3, 0.2]
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.patience      = patience
        self.use_smote     = use_smote
        self.use_focal_loss = use_focal_loss
        self.focal_gamma   = focal_gamma
        self.warmup_epochs = warmup_epochs
        self.val_fraction  = val_fraction

    def fit(self, X, y):
        X_arr = X if isinstance(X, np.ndarray) else np.array(X)
        y_arr = np.array(y, dtype=np.float32)

        if self.use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                X_arr, y_arr = SMOTE(random_state=42).fit_resample(X_arr, y_arr)
            except ImportError:
                pass

        # Split interno para early stopping por ROC-AUC
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_arr, y_arr, test_size=self.val_fraction, random_state=42, stratify=y_arr
        )

        X_t   = torch.tensor(X_tr,  dtype=torch.float32)
        y_t   = torch.tensor(y_tr,  dtype=torch.float32)
        X_v   = torch.tensor(X_val, dtype=torch.float32)
        y_v_np = y_val

        # Loss function
        if self.use_focal_loss:
            criterion = FocalLoss(gamma=self.focal_gamma)
        else:
            n_neg = float((y_tr == 0).sum())
            n_pos = float((y_tr == 1).sum())
            pos_w = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        self.model_ = ChurnNet(input_dim=X_t.shape[1], dropout=self.dropout)
        optimizer   = optim.AdamW(self.model_.parameters(),
                                  lr=self.lr, weight_decay=self.weight_decay)

        # Cosine annealing com warmup linear
        warmup = self.warmup_epochs
        cosine_epochs = max(self.epochs - warmup, 1)

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / cosine_epochs
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=self.batch_size, shuffle=True)

        best_auc, best_state, wait = 0.0, None, 0
        self.history_ = []

        for _ in range(self.epochs):
            self.model_.train()
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(X_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            # Early stopping por ROC-AUC no split de validação
            self.model_.eval()
            with torch.no_grad():
                val_proba = torch.sigmoid(self.model_(X_v)).numpy()
            val_auc = roc_auc_score(y_v_np, val_proba)
            self.history_.append(val_auc)

            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                wait       = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_arr = X if isinstance(X, np.ndarray) else np.array(X)
        X_t   = torch.tensor(X_arr, dtype=torch.float32)
        with torch.no_grad():
            proba_pos = torch.sigmoid(self.model_(X_t)).numpy()
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
