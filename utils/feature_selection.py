"""Análise e seleção de features — 6 métodos com ranking agregado."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import (
    SelectKBest, chi2, mutual_info_classif, f_classif, RFE
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

try:
    from IPython.display import display as _display
except ImportError:
    _display = print


def _normalizar(s: pd.Series) -> pd.Series:
    """Min-max normalização de uma Series para [0, 1]."""
    return (s - s.min()) / (s.max() - s.min() + 1e-10)


def analisar_features(
    X_train_enc: pd.DataFrame,
    y_train,
    k: int = 30,
    rf_importances: pd.Series = None,
    plot: bool = True,
):
    """
    Compara até 6 métodos de seleção de features e retorna ranking agregado.

    Métodos utilizados:
      1. ANOVA F-score (SelectKBest + f_classif)
      2. Chi-quadrado (SelectKBest + chi2)
      3. Informação Mútua (mutual_info_classif)
      4. Correlação de Pearson com o target
      5. RFE com Regressão Logística
      6. Feature Importance do Random Forest (opcional, via rf_importances)

    Parâmetros
    ----------
    X_train_enc    : pd.DataFrame — features já encodadas/transformadas (sem NaN)
    y_train        : array-like   — target de treino (binário: 0/1)
    k              : int          — número de features a selecionar no ranking final
    rf_importances : pd.Series, opcional
                     Importâncias do RF com index igual às colunas de X_train_enc.
                     Se fornecido, adiciona o método 6 ao ranking.
    plot           : bool — se True, exibe gráficos (ANOVA, Info. Mútua, Ranking)

    Retorna
    -------
    top_k           : list[str]   — nomes das top-k features pelo ranking agregado
    ranking_agregado: pd.Series   — scores normalizados somados (todas as features)

    Exemplo
    -------
    >>> top_k, ranking = analisar_features(X_tr_enc, y_train, k=30, plot=True)
    >>> top_k, ranking = analisar_features(
    ...     X_tr_enc, y_train, k=30,
    ...     rf_importances=rf_pipe.named_steps['classifier'].feature_importances_
    ... )
    """
    feat_names = X_train_enc.columns.tolist()
    y_arr = np.array(y_train)

    # MinMaxScaler — necessário para chi2 (requer X >= 0)
    scaler_mm = MinMaxScaler()
    X_scaled  = pd.DataFrame(
        scaler_mm.fit_transform(X_train_enc), columns=feat_names
    )

    # ── 1. ANOVA F-score ──────────────────────────────────────────────────────
    skb_f = SelectKBest(score_func=f_classif, k=k)
    skb_f.fit(X_scaled, y_arr)
    anova_scores = pd.Series(skb_f.scores_, index=feat_names).sort_values(ascending=False)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=anova_scores.head(k).values, y=anova_scores.head(k).index,
                    palette='Blues_d', ax=ax)
        ax.set_title('SelectKBest — ANOVA F-score')
        ax.set_xlabel('F-score')
        plt.tight_layout()
        plt.show()

    # ── 2. Chi-quadrado ───────────────────────────────────────────────────────
    skb_chi2 = SelectKBest(score_func=chi2, k=k)
    skb_chi2.fit(X_scaled, y_arr)
    chi2_scores = pd.Series(skb_chi2.scores_, index=feat_names).sort_values(ascending=False)

    # ── 3. Informação Mútua ───────────────────────────────────────────────────
    mi = mutual_info_classif(X_scaled, y_arr, random_state=42)
    mi_scores = pd.Series(mi, index=feat_names).sort_values(ascending=False)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=mi_scores.head(k).values, y=mi_scores.head(k).index,
                    palette='viridis', ax=ax)
        ax.set_title('Informação Mútua com o Target')
        ax.set_xlabel('Score')
        plt.tight_layout()
        plt.show()

    # ── 4. Correlação de Pearson ──────────────────────────────────────────────
    corr_target = (
        X_train_enc
        .reset_index(drop=True)
        .corrwith(pd.Series(y_arr))
        .abs()
        .sort_values(ascending=False)
    )

    # ── 5. RFE com Regressão Logística ────────────────────────────────────────
    lr_rfe = LogisticRegression(max_iter=1000, random_state=42)
    rfe = RFE(estimator=lr_rfe, n_features_to_select=k)
    rfe.fit(X_scaled, y_arr)
    rfe_ranking = pd.Series(rfe.ranking_, index=feat_names).sort_values()

    # ── Construir ranking agregado ────────────────────────────────────────────
    scores = [
        _normalizar(anova_scores),
        _normalizar(chi2_scores),
        _normalizar(mi_scores),
        _normalizar(corr_target),
        _normalizar(rfe_ranking.max() - rfe_ranking),  # menor rank = melhor
    ]

    n_metodos = len(scores)
    if rf_importances is not None:
        # ── 6. Feature Importance do RF (opcional) ────────────────────────────
        rf_scores = rf_importances.reindex(feat_names).fillna(0)
        scores.append(_normalizar(rf_scores))
        n_metodos += 1

    ranking_agregado = sum(scores).sort_values(ascending=False)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x=ranking_agregado.head(k).values,
            y=ranking_agregado.head(k).index,
            palette='rocket', ax=ax,
        )
        ax.set_title(f'Ranking Agregado ({n_metodos} métodos) — Top {k} features')
        ax.set_xlabel('Score Agregado (normalizado)')
        plt.tight_layout()
        plt.show()

    top_k = ranking_agregado.head(k).index.tolist()
    print(f"\n✔ Top {k} features pelo ranking agregado ({n_metodos} métodos):")
    print(top_k)
    return top_k, ranking_agregado
