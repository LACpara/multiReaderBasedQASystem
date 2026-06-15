import re

from pyexpat import model
from typing import Any, Literal, get_args, get_origin
from pathlib import Path
from logging import getLogger
from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined

from prompt_manager.type_resolver import TypeResolver


class SchemaParser:
    @classmethod
    def _build_list_type(cls, model_name: str, item_def: dict) -> type:
        """
        递归构建嵌套列表类型。
        处理任意深度的 list[list[list[...]]] 结构。
        """
        item_type_hint = item_def.get("_type", "")

        if item_type_hint.startswith("list"):
            # 递归处理嵌套列表
            nested_item_def = item_def.get("_item", {})
            inner_type = cls._build_list_type(model_name, nested_item_def)
            return list[inner_type]
        else:
            # 非列表类型，构建对应的模型
            # 过滤掉以 "_" 开头的元数据字段，只保留实际字段定义
            actual_fields = {
                k: v for k, v in item_def.items()
                if not k.startswith("_")
            }
            if actual_fields:
                # 有实际字段定义，构建 BaseModel
                return cls.build_model(model_name, actual_fields)
            elif item_type_hint:
                # 只有类型定义，解析为基本类型
                return TypeResolver.resolve_type(item_type_hint)
            else:
                # 空定义，返回 Any
                return Any

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
                    item_type = cls._build_list_type(
                        f"{model_name}_{key}".title(),
                        item_def
                    )
                    field_type = list[item_type]

            fields[key] = (
                field_type,
                Field(default=default, description=desc),
            )
        return create_model(model_name, **fields)

    @classmethod
    def _dump_list_item(cls, item_anno: type) -> dict:
        """
        递归处理嵌套列表元素的 dump。
        处理任意深度的 list[list[list[...]]] 结构。
        """
        if get_origin(item_anno) is list:
            # 递归处理嵌套列表
            inner_item_anno = get_args(item_anno)[0]
            return {
                "_type": "list",
                "_item": cls._dump_list_item(inner_item_anno)
            }
        elif isinstance(item_anno, type) and issubclass(item_anno, BaseModel):
            # BaseModel 类型，直接返回字段定义（不包含 _type）
            return cls.dump_model(item_anno)
        else:
            # 基本类型
            return {
                "_type": cls.annotation_2_str(item_anno),
                "_desc": ""
            }

    @classmethod
    def dump_model(cls, model: type[BaseModel]) -> dict[str, Any]:
        """
        将动态生成的 Pydantic 模型还原为 schema 定义。
        """
        schema = {}
        for name, field in model.model_fields.items():
            anno = field.annotation
            desc = field.description
            default_val = field.default
            is_required = default_val is PydanticUndefined

            field_def = {}

            if isinstance(anno, type) and issubclass(anno, BaseModel):
                field_def["_type"] = "dict"
                field_def.update(cls.dump_model(anno))

            elif get_origin(anno) is list:
                field_def["_type"] = "list"
                item_anno = get_args(anno)[0]
                field_def["_item"] = cls._dump_list_item(item_anno)

            else:
                field_def["_type"] = cls.annotation_2_str(anno)

            if not is_required:
                field_def["_default"] = default_val

            field_def["_desc"] = desc

            schema[name] = field_def
        return schema

    @classmethod
    def annotation_2_str(cls, anno: type) -> str:
        str_anno = str(anno)
        m = re.match(r"<class \'(.*)\'>", str_anno)
        if m:
            return m.group(1)
        elif str_anno.startswith("typing.") or str_anno.startswith("typing_extensions."):
            return str_anno.split(".", 1)[1]
        return str_anno