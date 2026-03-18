from abc import ABC, abstractmethod

from src.schemas.inference import InferenceResponse


class InferenceServiceInterface(ABC):
    """Contrato que todo serviço de inferência deve cumprir.

    Ao adicionar o modelo de ML, basta criar uma nova implementação
    desta interface sem alterar routers ou demais camadas (DIP/OCP).
    """

    @abstractmethod
    def predict(self) -> InferenceResponse:
        """Executa uma inferência e retorna a resposta padronizada."""


class HelloWorldInferenceService(InferenceServiceInterface):
    """Implementação stub — será substituída pelo serviço de ML real."""

    def predict(self) -> InferenceResponse:
        return InferenceResponse(message="Hello, World!")
