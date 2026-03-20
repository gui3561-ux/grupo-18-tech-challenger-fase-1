"""Funções de avaliação e comparação de modelos de ML."""

import os
import pathlib
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

try:
    from IPython.display import display as _display
except ImportError:
    _display = print


def avaliar_modelo(nome, estimador, X_test, y_test):
    """
    Avalia um estimador sklearn já treinado e retorna dict de métricas.

    Parâmetros
    ----------
    nome      : str — rótulo do modelo
    estimador : sklearn estimator — modelo já treinado (ex: best_estimator_ do Search)
    X_test    : array-like — features de teste
    y_test    : array-like — target real

    Retorna
    -------
    dict com Modelo, Accuracy, Precision, Recall, F1, ROC-AUC
    """
    y_pred  = estimador.predict(X_test)
    y_proba = estimador.predict_proba(X_test)[:, 1]
    return {
        'Modelo':     nome,
        'Accuracy':   round(accuracy_score(y_test, y_pred),  4),
        'Precision':  round(precision_score(y_test, y_pred), 4),
        'Recall':     round(recall_score(y_test, y_pred),    4),
        'F1':         round(f1_score(y_test, y_pred),        4),
        'ROC-AUC':    round(roc_auc_score(y_test, y_proba),  4),
    }


def comparar_modelos(X_train, X_test, y_train, y_test,
                     modelos, n_iter=60, cv=5, random_state=42,
                     mostrar_tabela=True,
                     salvar_modelos=True,
                     models_dir='../models',
                     experiment_name='churn_mvp',
                     tracking_uri='../mlflow'):
    """
    Treina e compara múltiplos Pipelines sklearn via RandomizedSearchCV.

    Cada Pipeline no dicionário `modelos` encapsula seu próprio pré-processamento
    (encoding + scaler + seleção de features), recebendo os dados brutos diretamente.
    Opcionalmente salva cada best_estimator_ como .pkl e registra no MLflow.

    Parâmetros
    ----------
    X_train, X_test  : pd.DataFrame — dados brutos (sem encoding manual)
    y_train, y_test  : pd.Series
    modelos          : dict — {nome: (pipeline, param_grid)} ou
                              {nome: (pipeline, param_grid, n_jobs)}
                              n_jobs por modelo — use 1 para PyTorch (evita conflito
                              entre multiprocessing do joblib e threads do PyTorch)
    n_iter           : int  — combinações testadas por modelo no RandomizedSearchCV
    cv               : int  — número de folds do StratifiedKFold
    random_state     : int
    mostrar_tabela   : bool — se True, exibe tabela colorida no Jupyter
    salvar_modelos   : bool — se True, salva .pkl em models_dir e loga no MLflow
    models_dir       : str  — caminho para a pasta de modelos (relativo ao notebook)
    experiment_name  : str  — nome do experimento MLflow
    tracking_uri     : str  — URI do MLflow tracking server (local file store)

    Retorna
    -------
    pd.DataFrame com métricas de todos os modelos, ordenado por ROC-AUC

    Exemplo
    -------
    >>> df = comparar_modelos(X_train, X_test, y_train, y_test, modelos_pipeline)
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    resultados  = []

    def _roc_auc_scorer(estimator, X, y):
        """Scorer direto para evitar o check is_regressor() do sklearn 1.4+."""
        return roc_auc_score(y, estimator.predict_proba(X)[:, 1])

    for nome, model_def in modelos.items():
        pipeline, params = model_def[0], model_def[1]
        n_jobs_model = model_def[2] if len(model_def) > 2 else -1
        print(f'\n>>> {nome}...')
        search = RandomizedSearchCV(
            pipeline, params,
            n_iter=n_iter, scoring=_roc_auc_scorer,
            cv=cv_strategy, random_state=random_state,
            n_jobs=n_jobs_model, error_score='raise',
        )
        search.fit(X_train, y_train)
        resultado = avaliar_modelo(nome, search.best_estimator_, X_test, y_test)
        resultado['Melhores Params'] = search.best_params_

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=nome):
            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics({
                'accuracy':   resultado['Accuracy'],
                'precision':  resultado['Precision'],
                'recall':     resultado['Recall'],
                'f1':         resultado['F1'],
                'roc_auc':    resultado['ROC-AUC'],
                'cv_roc_auc': round(search.best_score_, 4),
            })
            if salvar_modelos:
                pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
                base_name = nome.lower().replace(' ', '_')

                # Pipeline completo: recebe todas as colunas brutas → predição
                # Uso na API: pipeline.predict(X_raw) ou pipeline.predict_proba(X_raw)
                pipeline_path = os.path.join(models_dir, f'{base_name}_pipeline.pkl')
                with open(pipeline_path, 'wb') as f:
                    pickle.dump(search.best_estimator_, f)
                mlflow.log_artifact(pipeline_path, artifact_path='models')

            import warnings as _w
            with _w.catch_warnings():
                _w.filterwarnings('ignore', message='.*pickle.*')
                _w.filterwarnings('ignore', message='.*cloudpickle.*')
                mlflow.sklearn.log_model(search.best_estimator_, name='sklearn_model')

        resultados.append(resultado)
        print(f"    ROC-AUC CV : {search.best_score_:.4f}")
        print(f"    ROC-AUC    : {resultado['ROC-AUC']:.4f}")
        print(f"    Params     : {search.best_params_}")

    df_result = (
        pd.DataFrame(resultados)
        .sort_values('ROC-AUC', ascending=False)
        .reset_index(drop=True)
    )

    if mostrar_tabela:
        print('\n' + '=' * 60)
        print('  COMPARATIVO FINAL — TODOS OS MODELOS')
        print('=' * 60)
        _display(
            df_result.drop(columns='Melhores Params')
            .style.background_gradient(
                subset=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
                cmap='RdYlGn', vmin=0, vmax=1
            )
        )
    return df_result
