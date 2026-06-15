import ast

from pyexpat import model
from typing import Any, Literal, TypeAlias
from pathlib import Path
from logging import getLogger
from pydantic import BaseModel, Field, create_model

from prompt_manager.type_resolver import TypeResolver


class SchemaParser:
    @classmethod
    def build_model(cls, model_name: str, schema: dict) -> type[BaseModel]:
        fields = {}

        for key, value in schema.items():
            if key.startswith("_"): continue

            if not isinstance(value, dict):
                raise ValueError(f"Invalid schema field: {key}")

            type_hint = value.get("_type")
            if type_hint is None:
                raise ValueError(f"Missing type hint for field: {key}")

            field_type = TypeResolver.resolve_type(type_hint)
            desc = value.get("_desc", "")
            default = value.get("_default", ...)

            nested = {
                k: v
                for k, v in value.items()
                if not k.startswith("_")
            }

            if nested and field_type in (dict, dict[str, Any]):
                nested_model = cls.build_model(
                    f"{model_name}_{key}".title(),
                    nested
                )
                field_type = nested_model


            if type_hint.startswith("list"):
                if item_def := value.get("_item", None):
                    item_type = cls.build_model(
                        f"{model_name}_{key}".title(),
                        item_def
                    )
                    field_type: TypeAlias = list[item_type]

            fields[key] = (
                field_type,
                Field(default=default, description=desc),
            )
        return create_model(model_name, **fields)