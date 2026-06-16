import pydantic

from prompt_manager.domain import PromptDefinition
from pydantic import BaseModel, Field
import pytest
import json
import re


class TestPromptDefinition:
    def test_field(self):
        """测试 PromptDefinition 的字段属性"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"name": str, "age": int}}
        )

        prompt_def = PromptDefinition(
            meta={"version": "1.0"},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="Hello ${name}, you are ${age} years old."
        )

        assert prompt_def.meta == {"version": "1.0"}
        assert prompt_def.param_model == param_model
        assert prompt_def.output_model is None
        assert prompt_def.output_mode == "text"
        assert prompt_def.template == "Hello ${name}, you are ${age} years old."

    def test_render(self):
        """测试基本模板渲染功能"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"name": str, "age": int}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="Hello ${name}, you are ${age} years old."
        )

        result = prompt_def.render(name="Alice", age=30)
        assert result == "Hello Alice, you are 30 years old."

    def test_render_json(self):
        """测试 JSON 输出模式的渲染（包含输出约束）"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"query": str}}
        )

        output_model = type(
            "TestOutput",
            (BaseModel,),
            {"__annotations__": {"answer": str, "confidence": float}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=output_model,
            output_mode="json",
            template="Please answer: ${query}"
        )

        result = prompt_def.render(query="What is AI?")
        
        # 验证模板被正确渲染
        assert "Please answer: What is AI?" in result

        # 验证输出约束被添加
        m = re.search(r"```json\n(.*)```", result, flags=re.MULTILINE | re.DOTALL)
        assert m is not None
        output_schema = m.group(1)
        assert "answer" in output_schema
        assert "confidence" in output_schema

    def test_render_with_dict_param(self):
        """测试字典类型参数的渲染"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"user": dict}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="User info: ${user}"
        )

        user_data = {"name": "Bob", "age": 25}
        result = prompt_def.render(user=user_data)
        
        # 字典应该被序列化为 JSON
        assert '"name": "Bob"' in result or "'name': 'Bob'" in result

    def test_render_with_list_param(self):
        """测试列表类型参数的渲染"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"items": list}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="Items: ${items}"
        )

        items = ["apple", "banana", "cherry"]
        result = prompt_def.render(items=items)
        
        # 列表应该被序列化为 JSON
        assert "apple" in result
        assert "banana" in result
        assert "cherry" in result

    def test_render_missing_param(self):
        """测试缺少参数时抛出异常"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"name": str, "age": int}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="Hello ${name}, you are ${age} years old."
        )

        # 只提供 name，不提供 age
        # Pydantic 会抛出 ValidationError
        with pytest.raises(pydantic.ValidationError):  # Pydantic 验证错误
            prompt_def.render(name="Alice")

    def test_parse_output_text(self):
        """测试文本输出模式的解析"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="Test"
        )

        text = "This is a plain text response."
        result = prompt_def.parse_output(text)
        assert result == text

    def test_parse_output_json(self):
        """测试 JSON 输出模式的解析（无模型验证）"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="json",
            template="Test"
        )

        json_text = '{"answer": "42", "confidence": 0.95}'
        result = prompt_def.parse_output(json_text)
        assert result == {"answer": "42", "confidence": 0.95}

    def test_parse_output_with_model(self):
        """测试带模型的输出解析"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {}}
        )

        output_model = type(
            "TestOutput",
            (BaseModel,),
            {"__annotations__": {"answer": str, "confidence": float}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=output_model,
            output_mode="json",
            template="Test"
        )

        json_text = '{"answer": "42", "confidence": 0.95}'
        result = prompt_def.parse_output(json_text)
        
        # 结果应该是 output_model 的实例
        assert isinstance(result, output_model)
        assert result.answer == "42"
        assert result.confidence == 0.95

    def test_parse_output_invalid_json(self):
        """测试无效 JSON 解析"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="json",
            template="Test"
        )

        invalid_json = "This is not a valid JSON"
        with pytest.raises(ValueError, match="Invalid JSON format"):
            prompt_def.parse_output(invalid_json)

    def test_parse_output_malformed_json(self):
        """测试格式错误的 JSON 解析"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="json",
            template="Test"
        )

        malformed_json = '{"answer": "42", "confidence": 0.95'  # 缺少闭合括号
        with pytest.raises(ValueError, match="Invalid JSON format"):
            prompt_def.parse_output(malformed_json)

    def test_append_output_constraint(self):
        """测试输出约束的添加"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"query": str}}
        )

        output_model = type(
            "TestOutput",
            (BaseModel,),
            {"__annotations__": {"result": str}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=output_model,
            output_mode="json",
            template="Query: ${query}"
        )

        result = prompt_def.render(query="test")
        
        # 验证输出约束被正确添加
        assert "## 输出约束" in result
        assert "返回结果严格遵循以下 JSON Schema：" in result
        assert "```json" in result
        assert "result" in result

    def test_no_output_model(self):
        """测试无输出模型时不应添加输出约束"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"query": str}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="json",
            template="Query: ${query}"
        )

        result = prompt_def.render(query="test")
        
        # 验证输出约束未被添加
        assert "## 输出约束" not in result
        assert "```json" not in result

    def test_render_with_nested_dict(self):
        """测试嵌套字典的渲染"""
        database_model = type(
            "DataBaseParam",
            (BaseModel,),
            {"__annotations__": {"host": str, "port": int}}
        )

        cache_model = type(
            "CacheParam",
            (BaseModel,),
            {"__annotations__": {"enabled": bool, "ttl": int}}
        )

        config_model = type(
            "ConfigParam",
            (BaseModel,),
            {"__annotations__": {"database": database_model, "cache": cache_model}}
        )

        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"config": config_model}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="Config: ${config}"
        )

        config = {
            "database": {
                "host": "localhost",
                "port": 5432
            },
            "cache": {
                "enabled": True,
                "ttl": 3600
            }
        }
        result = prompt_def.render(config=config)
        
        # 验证嵌套结构被正确序列化
        assert "database" in result
        assert "localhost" in result
        assert "5432" in result
        assert "database" in result
        assert "cache" in result
        assert "enabled" in result
        assert "ttl" in result

    def test_render_with_multiple_placeholders(self):
        """测试多个占位符的渲染"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {
                "name": str,
                "age": int,
                "city": str,
                "country": str
            }}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="${name} is ${age} years old, lives in ${city}, ${country}"
        )

        result = prompt_def.render(
            name="Alice",
            age=30,
            city="Paris",
            country="France"
        )
        assert result == "Alice is 30 years old, lives in Paris, France"

    def test_render_with_special_characters(self):
        """测试包含特殊字符的参数渲染"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"text": str}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="Text: ${text}"
        )

        text = "Hello, world! @#$%^&*()"
        result = prompt_def.render(text=text)
        assert text in result

    def test_parse_output_with_validation_error(self):
        """测试输出验证失败的情况"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {}}
        )

        output_model = type(
            "TestOutput",
            (BaseModel,),
            {"__annotations__": {"age": int}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=output_model,
            output_mode="json",
            template="Test"
        )

        # 提供字符串而不是整数
        json_text = '{"age": "not_a_number"}'
        with pytest.raises(Exception):  # Pydantic 验证错误
            prompt_def.parse_output(json_text)

    def test_render_with_empty_template(self):
        """测试空模板的渲染"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template=""
        )

        result = prompt_def.render()
        assert result == ""

    def test_render_with_no_placeholders(self):
        """测试无占位符的模板渲染"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="This is a static template with no placeholders."
        )

        result = prompt_def.render()
        assert result == "This is a static template with no placeholders."

    def test_render_with_unicode(self):
        """测试 Unicode 字符的渲染"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {"text": str}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="text",
            template="Text: ${text}"
        )

        text = "你好世界 🌍"
        result = prompt_def.render(text=text)
        assert text in result

    def test_parse_output_with_nested_json(self):
        """测试嵌套 JSON 的解析"""
        param_model = type(
            "TestParams",
            (BaseModel,),
            {"__annotations__": {}}
        )

        prompt_def = PromptDefinition(
            meta={},
            param_model=param_model,
            output_model=None,
            output_mode="json",
            template="Test"
        )

        nested_json = {
            "user": {
                "name": "Alice",
                "age": 30
            },
            "items": ["item1", "item2"]
        }
        json_text = json.dumps(nested_json)
        result = prompt_def.parse_output(json_text)
        assert result == nested_json