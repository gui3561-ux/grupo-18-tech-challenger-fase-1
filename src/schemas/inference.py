from pydantic import BaseModel


class InferenceResponse(BaseModel):
    message: str

    model_config = {"json_schema_extra": {"example": {"message": "Hello, World!"}}}
