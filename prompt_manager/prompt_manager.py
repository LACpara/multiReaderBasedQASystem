# implement by chatgpt web agent
from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, create_model





class PromptManagerError(Exception): ...


class SchemaParser:
    BASIC_TYPES = {
        "str": str,
        "int": int,
        "float": float,
        "Any": Any,
    }

    @classmethod
    def parse_type(cls, type_hint: str) -> Any:
        type_hint = type_hint.strip()

        if "|" in type_hint:
            parts = [cls.parse_type(x.strip()) for x in type_hint.split("|")]
            result = parts[0]
            for p in parts[1:]:
                result = result | p
            return result

        if type_hint.startswith("list"):
            inner = Any
            m = re.match(r"list\[(.+)\]", type_hint)
            if m:
                inner = cls.parse_type(m.group(1))
            return list[inner]

        if type_hint.startswith("dict"):
            m = re.match(r"dict\[(.+),(.+)\]", type_hint)
            if m:
                k = cls.parse_type(m.group(1).strip())
                v = cls.parse_type(m.group(2).strip())
                return dict[k, v]
            return dict[str, Any]

        if type_hint.startswith("literal["):
            values = re.findall(r'["\']([^"]+)["\']', type_hint)
            from typing import Literal
            return Literal.__getitem__(tuple(values))

        if type_hint in cls.BASIC_TYPES:
            return cls.BASIC_TYPES[type_hint]

        raise PromptManagerError(f"Unsupported type: {type_hint}")

    @classmethod
    def build_model(cls, model_name: str, schema: dict) -> type[BaseModel]:
        fields = {}

        for key, value in schema.items():
            if key.startswith("_"):
                continue

            if not isinstance(value, dict):
                raise PromptManagerError(f"Invalid schema field: {key}")

            field_type = cls.parse_type(value.get("_type", "Any"))
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

            fields[key] = (
                field_type,
                Field(default=default, description=desc),
            )

        return create_model(model_name, **fields)


