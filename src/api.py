import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import base64
from pathlib import Path
from .prompts import SYSTEM_PROMPT
from .response import GeneralResponse
import os
from openai import OpenAI


MODEL_RATES = {
    "gpt-4o":      (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read())



@dataclass
class BaseContent:
    text: Optional[str] = None
    image: Optional[str | Path] = None

    def to_dict(self):
        if self.text is None and self.image is None:
            raise ValueError("Either text or image should be provided")
        type = "text" if self.text else "image_url"

        if isinstance(self.image, Path):
            self.image = self.image.as_posix()

        if isinstance(self.image, str):
            self.image = encode_image(self.image).decode("utf-8")

        content = (
            self.text
            if type == "text" 
            else {
                "url": f"data:image/jpeg;base64,{self.image}",
            }
        )
        return {"type": type, type: content}

class PromptMessages:
    def __init__(self, use_system_message: bool = True):
        self._messages = []
        if use_system_message:
            self.reset_message(use_system_message=True)

    @property
    def messages(self):
        return self._messages

    def add_message(self, role="user", image: Optional[str | Path] = None, text: Optional[str] = None):
        contents = []
        if image:
            contents.append(BaseContent(image=image))
        if text:
            contents.append(BaseContent(text=text))
        self._add_message(role, contents)

    def _add_message(self, role: str, content: Union[BaseContent, List[BaseContent]]):
        if not isinstance(content, list):
            content = [content]

        self._messages.append({"role": role, "content": [c.to_dict() for c in content]})

    def reset_message(self, use_system_message=True):
        self._messages = []
        if use_system_message:
            self.add_message(role="system", text=SYSTEM_PROMPT)
        return self

@dataclass
class GPTCost:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0
    model_name: str = "gpt-4o"

    def __add__(self, other: "GPTCost"):
        return GPTCost(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost + other.cost,
            self.model_name,
        )

    def __radd__(self, other: "GPTCost"):
        return self.__add__(other)

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0
        return self

    @classmethod
    def from_gpt_results(cls, model_name: str, result: Any):
        pt = result.usage.prompt_tokens
        ct = result.usage.completion_tokens
        rates = MODEL_RATES.get(model_name, (0.0, 0.0))
        prompt_rate, completion_rate = rates
        cost = (pt / 1000) * prompt_rate + (ct / 1000) * completion_rate
        return cls(pt, ct, cost, model_name)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)

class GPTWrapper:
    def __init__(self, 
        model_name: str = "gpt-4o", 
        model_params: Dict[str, Any] = None, 
        response_format: Any = GeneralResponse,
    ):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"), 
            organization=os.getenv("ORGANIZATION"),
        )
        
        self.model_name = model_name
        self.params = model_params or {
            "model": model_name,
        }
        self.response_format = response_format
        self.total_cost = GPTCost(model_name=self.model_name)
        self.error_requests = []

    def add_cost(self, results: List[Any]):
        if not isinstance(results, list):
            results = [results]
        for result in results:
            self.total_cost += GPTCost.from_gpt_results(self.model_name, result)

    def show_cost(self):
        print(self.total_cost)

    def ask(self, image: Optional[str | Path] = None, text: Optional[str] = None, use_system_message: bool = True):
        msgs = PromptMessages(use_system_message)
        msgs.add_message(image=image, text=text)
        result = self.client.beta.chat.completions.parse(
            messages=msgs.messages, 
            response_format=self.response_format, 
            **self.params,
        )
        self.add_cost(result)
        return result.choices[0].message.parsed
