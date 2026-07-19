
from pydantic import BaseModel


class CodeResultModel(BaseModel):
    python_code: str