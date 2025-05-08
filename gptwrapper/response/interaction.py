from pydantic import BaseModel
from typing import List

class InteractionResponse(BaseModel):
    interactions: List[str]
