from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str

    model_config = {"json_schema_extra": {"example": {"status": "ok", "version": "1.0.0"}}}
