from cgi import parse_multipart
import os
import re

from string import Template
from logging import getLogger
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from venv import logger

from pydantic.functional_validators import InstanceOf

@dataclass
class PromptTemplate:
    name: str
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
        shortage = set(self.params)
        for param in params.keys():
            if param not in self.params:
                logger.error(f"Prompt file {self.name} varify params failure, with params: {params}, need: {self.params}")
                return False
            shortage.remove(param)
        if shortage:
            logger.error(f"Prompt file {self.name} varify params shortage: {shortage} with params: {params}")
            return False
        logger.debug(f"Prompt file {self.name} varify params: {params}")
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

                logger.debug(f"Prompt file {prompt_name} parse params: {params}")

                matched = re.search(r"@begin(.*)@end", content, re.S)

                if matched is None:
                    logger.error(f"Prompt file {prompt_name} could not extract template body, with content: {content}")
                    return None
                
                logger.debug(f"Prompt file {prompt_name} parse template body: {matched.group(1).strip()}")

                body = matched.group(1).strip()
                template = Template(body)
            result = self.prompts[prompt_name] = PromptTemplate(name=prompt_name, params=params, template=template)
        return result
    
    def get_prompt(self, prompt_name, **kwargs) -> str:
        """
        获取并渲染指定名称的提示模板。
        :param prompt_name: 提示模板名称。
        :param kwargs: 渲染提示模板的参数。
        :return: 渲染后的提示字符串。
        """
        prompt = self.load(prompt_name)
        if prompt is None:
            raise ValueError(f"prompt {prompt_name} not found")
        return prompt.render(**kwargs)
