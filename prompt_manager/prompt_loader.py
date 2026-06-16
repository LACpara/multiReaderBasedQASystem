from asyncio import protocols
import hashlib
import yaml
import re

from pathlib import Path
from functools import lru_cache

from prompt_manager.domain import PromptDefinition
from prompt_manager.schema_parser import SchemaParser


class PromptLoader:
    def __init__(self, base_dir: str | Path, recursive: bool = True):
        self.base_dir = Path(base_dir)
        self.prompts = {}
        self.recursive = recursive
        if not self.base_dir.is_dir():
            raise ValueError(f"The argument `base_dir` should be a direction")
        self._load_prompt_list(self.base_dir)

    def _load_prompt_list(self, root_dir: Path):
        for child in root_dir.iterdir():
            if child.is_dir() and self.recursive:
                self._load_prompt_list(child)
                continue
            
            if child.name.endswith((".prompt.yaml", ".prompt.md")):
                prompt_base_name, *_ = child.name.rsplit(".", 2)
                if prefix := root_dir.relative_to(self.base_dir).parts:
                    prompt_full_name = ".".join([*prefix, prompt_base_name])
                else:
                    prompt_full_name = prompt_base_name

                if prompt_full_name not in self.prompts:
                    self.prompts[prompt_full_name] = (
                        root_dir / (prompt_base_name + ".prompt.yaml"),
                        root_dir / (prompt_base_name + ".prompt.md")
                    )
    
    def list_all_prompt(self) -> list[str]:
        return list(self.prompts.keys())

    def get_prompt(self, prompt_name: str) -> PromptDefinition:
        if prompt_name not in self.prompts:
            raise ValueError(f"prompt {prompt_name} is not found")
        
        yaml_path, md_path = self.prompts[prompt_name]
        prompt_template = self.load(yaml_path, md_path)
        return prompt_template
        
    @lru_cache
    def load(self, yaml_path: str | Path, md_path: str | Path) -> PromptDefinition:
        yaml_path = Path(yaml_path)
        md_path = Path(md_path)

        yaml_text = yaml_path.read_text(encoding="utf-8")
        md_text = md_path.read_text(encoding="utf-8")

        config = yaml.safe_load(yaml_text)

        param_schema = config.get("param", {})
        output_cfg = config.get("output", {"mode": "text"})
        meta = {k:v for k, v in config.items() if k not in ("param", "output")}

        param_model = SchemaParser.build_model(
            "PromptParams",
            param_schema,
        )

        output_model = None

        if output_cfg.get("mode") == "json":
            output_model = SchemaParser.build_model(
                "PromptOutput",
                output_cfg["schema"],
            )

        return PromptDefinition(
            meta=meta,
            param_model=param_model,
            output_model=output_model,
            output_mode=output_cfg.get("mode", "text"),
            template=md_text,
        )