from cgi import parse_multipart
import os
import re

from string import Template
from logging import getLogger
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from pydantic.functional_validators import InstanceOf

@dataclass
class PromptTemplate:
    params: set[str]
    template: Template | None = None
    
    def render(self, raise_: bool = True, **params: dict[Any]) -> str:
        if not self._varify_params(params):
            message = f"render template failure with incompatible params: {params}, need: {self.params}"
            if raise_: raise ValueError(message)
            getLogger(__file__).warning(message)
            return False
        return self.template.safe_substitute(params)

    def _varify_params(self, params: dict[Any]) -> bool:
        for param in params.keys():
            if param not in self.params:
                return False
        return True


class PromptLoader:
    def __init__(self, base_dir: Path, suffix: str = ".prompt", recursive: bool = True):
        self.base_dir = base_dir
        self.suffix = suffix
        self.recursive = recursive
        self.prompts: dict[str, PromptTemplate | Path] = {}
        self._load_promtp_list(self.base_dir)
        
    def _load_promtp_list(self, dir_: Path):
        for file_name in os.listdir(dir_):
            file_path = dir_ / file_name
            if file_path.is_dir() and self.recursive:
                self._load_promtp_list(file_path)
            elif file_path.suffix == self.suffix:
                prefixes = dir_.relative_to(self.base_dir).parts
                prompt_name = "/".join((*prefixes, file_path.stem)) if prefixes else file_path.stem
                self.prompts[prompt_name] = file_path
    
    def list_all(self) -> list[str]:
        return list(self.prompts.keys())

    def load(self, prompt_name) -> PromptTemplate | None:
        if prompt_name not in self.prompts: return None
        result = self.prompts[prompt_name]
        if isinstance(result, Path):
            with open(result, "r", encoding="utf-8") as fp:
                content = fp.read()
                params = set(re.findall(r"^@param:\s*(\w+)", content, re.M))
                matched = re.search(r"@begin(.*)@end", content, re.S)

                if matched is None:
                    return None

                body = matched.group(1).strip()
                template = Template(body)
            result = self.prompts[prompt_name] = PromptTemplate(params=params, template=template)
        return result
