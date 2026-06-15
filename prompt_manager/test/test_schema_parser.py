from re import M

import pydantic
from pydantic_core import SchemaError
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
        
        print(instance.model_dump_json(indent=2))
        pprint(model.model_json_schema(), indent=2)


class TestSchemaParserResverse:
    def test_simple(self):
        schema = {
            "simple_para": {
                "_type": "int",
                "_desc": "the simple parameter's description",
                "_default": 1314
            }
        }

        model = SchemaParser.build_model("simple_test", schema)
        assert issubclass(model, BaseModel)
        assert "simple_para" in model.model_fields

        field_info = model.model_fields["simple_para"]
        assert not field_info.is_required()
        assert field_info.annotation is int
        assert field_info.default == 1314
        assert field_info.description == "the simple parameter's description"

        instance = model()
        assert instance.simple_para == 1314

        with pytest.raises(pydantic.ValidationError):
            model(simple_para="hello")

        dumped = SchemaParser.dump_model(model)

        assert isinstance(dumped, dict)
        assert "simple_para" in dumped

        para_info = dumped["simple_para"]
        assert "_type" in para_info
        assert para_info["_type"] == "int"
        assert "_desc" in para_info
        assert para_info["_desc"] == "the simple parameter's description"
        assert "_default" in para_info
        assert para_info["_default"] == 1314
    
    
    def test_list_case(self):
        schema = {
            "list_para": {
                "_type": "list",
                "_desc": "the list type parameters",
                "_item": {
                    "para_1": {
                        "_type": "int",
                        "_desc": "element 1",
                    },
                    "para_2": {
                        "_type": "str",
                        "_desc": "element 2"
                    }
                }
            }
        }

        model = SchemaParser.build_model("list_para", schema)
        assert issubclass(model, BaseModel)

        dumped = SchemaParser.dump_model(model)
        assert isinstance(dumped, dict)

        assert "list_para" in dumped
        para_info = dumped["list_para"]

        assert "_type" in para_info
        assert para_info["_type"] == "list"
        assert "_desc" in para_info
        assert para_info["_desc"] == "the list type parameters"
        assert "_default" not in para_info

        item_info = para_info["_item"]
        for name, item_def in schema["list_para"]["_item"].items():
            assert name in item_info
            
            sub_item_info = item_info[name]

            assert "_type" in sub_item_info
            assert sub_item_info["_type"] == item_def["_type"]

            assert "_desc" in sub_item_info
            assert sub_item_info["_desc"] == item_def["_desc"]

            assert "_default" not in item_def
    

    def test_nested_with_single_layer_case(self):
        schema = {
            "nested_para": {
                "_type": "dict",
                "_desc": "the nested parameter",
                "para_a": {
                    "_type": "int",
                    "_desc": "para a ~",
                },
                "para_b": {
                    "_type": "str",
                    "_desc": "para b ~",
                    "_default": "hello world"
                }
            }
        }

        model = SchemaParser.build_model("nested_para", schema)
        assert issubclass(model, BaseModel)

        dumped = SchemaParser.dump_model(model)
        assert isinstance(dumped, dict)

        assert "nested_para" in dumped
        para_info = dumped["nested_para"]

        assert "_type" in para_info
        assert para_info["_type"] == "dict"
        assert "_desc" in para_info
        assert para_info["_desc"] == "the nested parameter"
        assert "_default" not in para_info

        assert "para_a" in para_info
        nested_para_a_info = para_info["para_a"]
        assert "para_b" in para_info
        nested_para_b_info = para_info["para_b"]

        assert "_type" in nested_para_a_info
        assert nested_para_a_info["_type"] == "int"
        assert "_desc" in nested_para_a_info
        assert nested_para_a_info["_desc"] == "para a ~"
        assert "_default" not in nested_para_a_info

        assert "_type" in nested_para_b_info
        assert nested_para_b_info["_type"] == "str"
        assert "_desc" in nested_para_b_info
        assert nested_para_b_info["_desc"] == "para b ~"
        assert "_default" in nested_para_b_info
        assert nested_para_b_info["_default"] == "hello world"

    def test_multiple_basic_types(self):
        """测试多种基本类型的 schema"""
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
            "score": {
                "_type": "float",
                "_desc": "User score",
                "_default": 95.5,
            },
        }

        model = SchemaParser.build_model("MultipleTypes", schema)
        assert issubclass(model, BaseModel)

        dumped = SchemaParser.dump_model(model)
        assert isinstance(dumped, dict)
        assert len(dumped) == 4

        # 测试字符串类型
        assert dumped["name"]["_type"] == "str"
        assert dumped["name"]["_desc"] == "User name"
        assert dumped["name"]["_default"] == "Anonymous"

        # 测试整数类型（无默认值）
        assert dumped["age"]["_type"] == "int"
        assert dumped["age"]["_desc"] == "User age"
        assert "_default" not in dumped["age"]

        # 测试布尔类型
        assert dumped["is_active"]["_type"] == "bool"
        assert dumped["is_active"]["_desc"] == "Active status"
        assert dumped["is_active"]["_default"] == True

        # 测试浮点数类型
        assert dumped["score"]["_type"] == "float"
        assert dumped["score"]["_desc"] == "User score"
        assert dumped["score"]["_default"] == 95.5

    def test_deeply_nested_schema(self):
        """测试深层嵌套的 schema 结构"""
        schema = {
            "outer": {
                "_type": "dict",
                "_desc": "Outer dict",
                "inner": {
                    "_type": "dict",
                    "_desc": "Inner dict",
                    "value": {
                        "_type": "str",
                        "_desc": "Deep value",
                        "_default": "deep",
                    },
                },
            },
        }

        model = SchemaParser.build_model("DeepNested", schema)
        assert issubclass(model, BaseModel)

        dumped = SchemaParser.dump_model(model)
        assert isinstance(dumped, dict)

        # 验证外层
        assert "outer" in dumped
        assert dumped["outer"]["_type"] == "dict"
        assert dumped["outer"]["_desc"] == "Outer dict"

        # 验证内层
        assert "inner" in dumped["outer"]
        assert dumped["outer"]["inner"]["_type"] == "dict"
        assert dumped["outer"]["inner"]["_desc"] == "Inner dict"

        # 验证深层值
        assert "value" in dumped["outer"]["inner"]
        assert dumped["outer"]["inner"]["value"]["_type"] == "str"
        assert dumped["outer"]["inner"]["value"]["_desc"] == "Deep value"
        assert dumped["outer"]["inner"]["value"]["_default"] == "deep"

    def test_list_of_list_schema(self):
        """测试嵌套列表类型（当前实现限制：内层列表的描述和最内层类型信息无法完全保留）"""
        schema = {
            "matrix": {
                "_type": "list",
                "_desc": "2D matrix",
                "_item": {
                    "_type": "list",
                    "_desc": "Row of numbers",
                    "_item": {
                        "_type": "int",
                        "_desc": "Single number",
                    },
                },
            },
        }

        model = SchemaParser.build_model("Matrix", schema)
        assert issubclass(model, BaseModel)

        dumped = SchemaParser.dump_model(model)
        assert isinstance(dumped, dict)

        # 验证外层列表
        assert "matrix" in dumped
        assert dumped["matrix"]["_type"] == "list"
        assert dumped["matrix"]["_desc"] == "2D matrix"

        # 验证内层列表（当前实现限制：内层列表的描述信息无法保留）
        assert "_item" in dumped["matrix"]
        assert dumped["matrix"]["_item"]["_type"] == "list"

        # 验证存在嵌套结构
        assert "_item" in dumped["matrix"]["_item"]

    def test_literal_type_dump(self):
        """测试 Literal 类型的 dump"""
        schema = {
            "status": {
                "_type": "Literal['active', 'inactive']",
                "_desc": "Status value",
                "_default": "active",
            },
        }

        model = SchemaParser.build_model("LiteralDumpTest", schema)
        assert issubclass(model, BaseModel)

        dumped = SchemaParser.dump_model(model)
        assert isinstance(dumped, dict)

        assert "status" in dumped
        assert dumped["status"]["_type"] == "Literal['active', 'inactive']"
        assert dumped["status"]["_desc"] == "Status value"
        assert dumped["status"]["_default"] == "active"

    def test_mixed_complex_types(self):
        """测试混合复杂类型的 schema"""
        schema = {
            "basic_str": {
                "_type": "str",
                "_desc": "Basic string",
            },
            "nested_dict": {
                "_type": "dict",
                "_desc": "Nested dictionary",
                "name": {
                    "_type": "str",
                    "_desc": "Name inside dict",
                    "_default": "test",
                },
            },
            "list_of_objects": {
                "_type": "list",
                "_desc": "List of objects",
                "_item": {
                    "id": {
                        "_type": "int",
                        "_desc": "Item ID",
                    },
                    "value": {
                        "_type": "str",
                        "_desc": "Item value",
                    },
                },
            },
        }

        model = SchemaParser.build_model("MixedTypes", schema)
        assert issubclass(model, BaseModel)

        dumped = SchemaParser.dump_model(model)
        assert isinstance(dumped, dict)
        assert len(dumped) == 3

        # 测试基本字符串
        assert dumped["basic_str"]["_type"] == "str"
        assert "_default" not in dumped["basic_str"]

        # 测试嵌套字典
        assert dumped["nested_dict"]["_type"] == "dict"
        assert "name" in dumped["nested_dict"]
        assert dumped["nested_dict"]["name"]["_default"] == "test"

        # 测试对象列表
        assert dumped["list_of_objects"]["_type"] == "list"
        assert "_item" in dumped["list_of_objects"]
        assert "id" in dumped["list_of_objects"]["_item"]
        assert "value" in dumped["list_of_objects"]["_item"]

    def test_empty_dict_schema(self):
        """测试空字典的 schema"""
        schema = {}

        model = SchemaParser.build_model("EmptyModel", schema)
        assert issubclass(model, BaseModel)

        dumped = SchemaParser.dump_model(model)
        assert isinstance(dumped, dict)
        assert len(dumped) == 0

    def test_dict_with_list_and_dict_mixed(self):
        """测试字典中同时包含列表和字典类型"""
        schema = {
            "data": {
                "_type": "dict",
                "_desc": "Mixed container",
                "items": {
                    "_type": "list",
                    "_desc": "List of items",
                    "_item": {
                        "name": {
                            "_type": "str",
                            "_desc": "Item name",
                        },
                    },
                },
                "metadata": {
                    "_type": "dict",
                    "_desc": "Metadata dict",
                    "version": {
                        "_type": "str",
                        "_desc": "Version string",
                        "_default": "1.0",
                    },
                },
            },
        }

        model = SchemaParser.build_model("MixedContainer", schema)
        assert issubclass(model, BaseModel)

        dumped = SchemaParser.dump_model(model)
        assert isinstance(dumped, dict)

        # 验证外层
        assert "data" in dumped
        assert dumped["data"]["_type"] == "dict"

        # 验证列表字段
        assert "items" in dumped["data"]
        assert dumped["data"]["items"]["_type"] == "list"
        assert "_item" in dumped["data"]["items"]

        # 验证嵌套字典字段
        assert "metadata" in dumped["data"]
        assert dumped["data"]["metadata"]["_type"] == "dict"
        assert "version" in dumped["data"]["metadata"]
        assert dumped["data"]["metadata"]["version"]["_default"] == "1.0"

