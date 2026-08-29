from typing import Any, Dict, List
from pydantic import BaseModel, Field

class MLPredictionRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(min_length=1)

class ProfilePredictionRequest(BaseModel):
    profile: Dict[str, Any]
