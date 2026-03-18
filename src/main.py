from fastapi import FastAPI

from src.api.v1.router import api_router

APP_TITLE = "Churn Prediction API"
APP_DESCRIPTION = (
    "API de inferência para o modelo de predição de churn de clientes. "
    "Construída com FastAPI seguindo princípios SOLID e pronta para receber "
    "um modelo de Machine Learning."
)
APP_VERSION = "1.0.0"


def create_app() -> FastAPI:
    """Factory da aplicação FastAPI."""
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(api_router)

    return app


app = create_app()
