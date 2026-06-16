from datetime import date
from pathlib import Path

from prompt_manager.prompt_loader import PromptLoader
from prompt_manager.domain import PromptDefinition

BASE_DIR = Path(__file__).resolve().parent.parent / "samples"

class TestPromptLoader:
    def test_load_prompt_name(self):
        loader = PromptLoader(BASE_DIR, recursive=False)
        prompts = loader.list_all_prompt()

        assert "demo1" in prompts
        assert "demo2" in prompts

    def test_load_prompt_name_with_recursive(self):
        loader = PromptLoader(BASE_DIR, recursive=True)
        prompts = loader.list_all_prompt()

        assert "demo1" in prompts
        assert "demo2" in prompts
        assert "inner.demo3" in prompts
    
    def test_load_path_valid(self):
        loader = PromptLoader(BASE_DIR, recursive=False)
        prompts: dict[str, tuple[Path, Path]] = loader.prompts

        demo1_yaml, demo1_md = prompts["demo1"]
        assert demo1_yaml.exists()
        assert demo1_md.exists()
        assert demo1_yaml.samefile(BASE_DIR / "demo1.prompt.yaml")
        assert demo1_md.samefile(BASE_DIR / "demo1.prompt.md")

        demo2_yaml, demo2_md = prompts["demo2"]
        assert demo2_yaml.exists()
        assert demo2_md.exists()
        assert demo2_yaml.samefile(BASE_DIR / "demo2.prompt.yaml")
        assert demo2_md.samefile(BASE_DIR / "demo2.prompt.md")
    
    def test_load_path_valid_with_recursive(self):
        loader = PromptLoader(BASE_DIR, recursive=True)
        prompts: dict[str, tuple[Path, Path]] = loader.prompts

        demo3_yaml, demo3_md = prompts["inner.demo3"]
        assert demo3_yaml.exists()
        assert demo3_md.exists()
        assert demo3_yaml.samefile(BASE_DIR / "inner" / "demo3.prompt.yaml")
        assert demo3_md.samefile(BASE_DIR / "inner" / "demo3.prompt.md")

    def test_load_promt_content(self):
        loader = PromptLoader(BASE_DIR, recursive=False)
        prompt_def = loader.get_prompt("demo1")

        assert isinstance(prompt_def, PromptDefinition)
        assert prompt_def.meta == {
            "title": "问候语生成示例",
            "author": "张三",
            "email": "zhang@example.com",
            "last_modified": date(2026, 6, 8),
            "version": 1.0,
            "tags": ["示例", "文本输出"],
            "description": "根据用户名和年龄，生成个性化问候语。"
        }
        assert prompt_def.output_mode == "text"
        assert "username" in prompt_def.param_model.model_fields
        assert "age" in prompt_def.param_model.model_fields

        assert prompt_def.param_model.model_fields["username"].annotation is str
        assert prompt_def.param_model.model_fields["username"].description == "用户姓名"
        assert prompt_def.param_model.model_fields["username"].is_required()

        assert prompt_def.param_model.model_fields["age"].annotation is int
        assert prompt_def.param_model.model_fields["age"].description == "用户年龄"
        assert not prompt_def.param_model.model_fields["age"].is_required()
        assert prompt_def.param_model.model_fields["age"].default == 18