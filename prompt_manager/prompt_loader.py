from pathlib import Path
import hashlib
import yaml
from prompt_manager.domain import PromptDefinition
from prompt_manager.schema_parser import SchemaParser

class PromptLoader:
    def __init__(self, cache_dir: str = ".prompt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def load(self, yaml_path: str | Path) -> PromptDefinition:
        yaml_path = Path(yaml_path)
        md_path = yaml_path.with_suffix(".prompt.md")

        yaml_text = yaml_path.read_text(encoding="utf-8")
        md_text = md_path.read_text(encoding="utf-8")

        cache_key = hashlib.sha256(
            (yaml_text + md_text).encode("utf-8")
        ).hexdigest()

        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            pass

        config = yaml.safe_load(yaml_text)

        param_schema = config.get("param", {})
        output_cfg = config.get("output", {"mode": "text"})
        meta = config.get("meta", {})

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