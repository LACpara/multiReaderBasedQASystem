from re import M

import pydantic
import pytest

from prompt_manager.schema_parser import SchemaParser
from pydantic import BaseModel, Field, ValidationError
from pprint import pprint
from typing import get_origin
from unittest.mock import patch, Mock

from typing import Any


class TestSchemaParser:
    def test_parse_simple_schema(self):
        schema = {
            "param_1": {
                "_type": "str",
                "_desc": "param_1 description",
                "_default": "default_value",
            }
        }

        model = SchemaParser.build_model(
            "PromptParams",
            schema,
        )

        # 测试模型名称
        assert model.__name__ == "PromptParams"
        
        # 测试模型继承自 BaseModel
        assert issubclass(model, BaseModel)
        
        # 测试字段数量
        assert len(model.model_fields) == 1
        
        # 测试字段名称存在
        assert "param_1" in model.model_fields
        
        # 获取字段信息
        field_info = model.model_fields["param_1"]
        
        # 测试字段类型注解
        assert field_info.annotation == str
        
        # 测试字段是否必填（有默认值则不是必填）
        assert field_info.is_required() == False
        
        # 测试字段默认值
        assert field_info.default == "default_value"
        
        # 测试字段描述
        assert field_info.description == "param_1 description"

    def test_parse_schema_with_multiple_fields(self):
        """测试包含多个字段的 schema"""
        schema = {
            "name": {
                "_type": "str",
                "_desc": "User name",
                "_default": "Anonymous",
            },
            "age": {
                "_type": "int",
                "_desc": "User age",
            },
            "is_active": {
                "_type": "bool",
                "_desc": "Active status",
                "_default": True,
            },
        }

        model = SchemaParser.build_model("UserParams", schema)
        
        # 测试字段数量
        assert len(model.model_fields) == 3
        
        # 测试各个字段
        assert model.model_fields["name"].annotation == str
        assert model.model_fields["age"].annotation == int
        assert model.model_fields["is_active"].annotation == bool
        
        # age 字段没有默认值，应该是必填的
        assert model.model_fields["age"].is_required() == True

    def test_model_instance_validation(self):
        """测试生成的模型能否正确实例化和验证数据"""
        schema = {
            "name": {
                "_type": "str",
                "_desc": "User name",
            },
            "age": {
                "_type": "int",
                "_desc": "User age",
                "_default": 18,
            },
        }

        model = SchemaParser.build_model("User", schema)
        
        # 测试正常实例化
        instance = model(name="John", age=25)
        assert instance.name == "John"
        assert instance.age == 25
        
        # 测试使用默认值
        instance2 = model(name="Jane")
        assert instance2.name == "Jane"
        assert instance2.age == 18
        
        # 测试类型验证 - 传入错误类型应该抛出异常
        with pytest.raises(ValidationError):
            model(name="John", age="not_a_number")

    def test_nested_schema(self):
        """测试嵌套的 schema 结构"""
        schema = {
            "user": {
                "_type": "dict",
                "_desc": "User information",
                "name": {
                    "_type": "str",
                    "_desc": "User name",
                },
                "age": {
                    "_type": "int",
                    "_desc": "User age",
                },
            },
            "count": {
                "_type": "int",
                "_desc": "Item count",
            },
        }

        model = SchemaParser.build_model("NestedParams", schema)
        
        # 测试字段存在
        assert "user" in model.model_fields
        assert "count" in model.model_fields
        
        # 测试嵌套模型类型（.title() 会将首字母大写，其他字母小写）
        user_model = model.model_fields["user"].annotation
        assert user_model.__name__ == "Nestedparams_User"
        assert issubclass(user_model, BaseModel)
        
        # 测试嵌套模型的字段
        assert "name" in user_model.model_fields
        assert "age" in user_model.model_fields
        
        # 测试实例化嵌套模型
        instance = model(user={"name": "John", "age": 25}, count=10)
        assert instance.user.name == "John"
        assert instance.user.age == 25
        assert instance.count == 10

    def test_schema_with_list_type(self):
        """测试包含列表类型的 schema"""
        schema = {
            "tags": {
                "_type": "list[str]",
                "_desc": "List of tags",
                "_default": [],
            },
        }

        model = SchemaParser.build_model("ListParams", schema)
        
        # 测试字段类型
        assert model.model_fields["tags"].annotation == list[str]
        
        # 测试实例化
        instance = model(tags=["tag1", "tag2"])
        assert instance.tags == ["tag1", "tag2"]

    def test_schema_with_literal_type(self):
        """测试包含 Literal 类型的 schema"""
        schema = {
            "status": {
                "_type": "Literal['active', 'inactive']",
                "_desc": "Status value",
                "_default": "active",
            },
        }

        model = SchemaParser.build_model("LiteralParams", schema)
        
        # 测试正常值
        instance = model(status="active")
        assert instance.status == "active"
        
        # 测试无效值应该抛出异常
        with pytest.raises(ValidationError):
            model(status="invalid")
        
    def test_schema_with_complex_generic_type(self):
        """测试包含复杂泛型类型的 schema"""
        schema = {
            "complex_para": {
                "_type": "list",
                "_desc": "complex generic type",
                "_item": {
                    "_type": "dict",
                    "_desc": "the second list element is dict",
                    "_default": dict(),
                    "para_1": {
                        "_type": "str",
                        "_desc": "the first dict property"
                    },
                    "para_2": {
                        "_type": "str",
                        "_desc": "the second dict property",
                    }
                }
            }
        }

        model = SchemaParser.build_model("ComplexityParams", schema)

        assert issubclass(model, BaseModel)
        assert model.__name__ == "ComplexityParams"

        assert "complex_para" in model.model_fields
        assert get_origin(model.model_fields["complex_para"].annotation) is list

        instance = model(complex_para=[{"para_1": "hello", "para_2": "world"}])
        assert instance.complex_para[0].para_1 == "hello"
        assert instance.complex_para[0].para_2 == "world"

        with pytest.raises(pydantic.ValidationError):
            model(complex_para=[{
                "para_1": 123,
                "para_2": "ok"
            }])
        # assert issubclass(model, BaseModel)
        # assert model.__name__ == "ComplexityParams"

        # assert "complex_para" in model.model_fields
        # assert model.model_fields["complex_para"].annotation == list[str | dict]
        # assert model.model_fields["complex_para"].description == "complex generic type"

        # assert "item_0" in model.model_fields["complex_para"].model_fields
        # assert "item_1" in model.model_fields["complex_para"].model_fields

        # assert model.model_fields["complex_para"].model_fields["item_0"].annotation == str
        # assert model.model_fields["complex_para"].model_fields["item_1"].annotation == dict[str, str]

        # assert model.model_fields["complex_para"].model_fields["item_1"].description == "the second list element is dict"
        # assert model.model_fields["complex_para"].model_fields["item_1"].default == dict()
        # assert model.model_fields["complex_para"].model_fields["item_1"].model_fields["para_1"].annotation == str

        # assert model.model_fields["complex_para"].model_fields["item_1"].model_fields["para_1"].description == "the first dict property"
        # assert model.model_fields["complex_para"].model_fields["item_1"].model_fields["para_2"].annotation == str

        # assert model.model_fields["complex_para"].model_fields["item_1"].model_fields["para_2"].description == "the second dict property"
        # assert model.model_fields["complex_para"].model_fields["item_1"].model_fields["para_2"].default == ""