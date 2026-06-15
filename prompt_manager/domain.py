from dataclasses import dataclass
import re
import json
from pydantic import BaseModel, Field
from typing import Any


PLACEHOLDER_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass
class PromptDefinition:
    meta: dict
    param_model: type[BaseModel]
    output_model: type[BaseModel] | None
    output_mode: str
    template: str

    def render(self, **kwargs) -> str:
        data = self.param_model(**kwargs).model_dump()

        def replace(match: re.Match) -> str:
            key = match.group(1)
            if key not in data:
                raise ValueError(f"Missing param: {key}")
            value = data[key]
            return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)

        prompt = PLACEHOLDER_RE.sub(replace, self.template)

        prompt = self._append_output_constraint(prompt)

        return prompt

    def _append_output_constraint(self, prompt: str) -> str:
        if self.output_mode == "json" and self.output_model:
            schema = self.output_model.model_json_schema()

            prompt += "\n".join(
                "\n\n## 输出约束",
                "返回结果严格遵循 JSON 格式：",
                "```json",
                f"{json.dumps(schema, ensure_ascii=False, indent=2)}",
                "```",
            )

        return prompt

    def parse_output(self, text: str) -> Any:
        if self.output_mode == "text":
            return text

        data = self._safe_parse_json(text)

        if self.output_model:
            return self.output_model.model_validate(data)

        return data

    def _safe_parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
        except Exception as e:
            raise ValueError(f"Error parsing JSON: {e}")