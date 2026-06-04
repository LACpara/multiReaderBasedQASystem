import pytest
from string import Template
from unittest.mock import patch, mock_open
from pathlib import Path
from pyfakefs.fake_filesystem_unittest import TestCase as FakeFsTestCase

from hmr.prompt_loader import PromptLoader
from hmr.prompt_loader import PromptTemplate


PROMPTS = {
    "prompt-1": "promt 01",
    "prompt-2": "promt 02",
    "prompt-3": "promt 03",
    "prompt-4": "promt 04",
}

def _list_equal(list1, list2):
    return all(item in list2 for item in list1) and not any(item not in list1 for item in list2)

class TestPromptLoader(FakeFsTestCase):
    def setUp(self):
        self.setUpPyfakefs()
        self.baseDir = Path("/usr/prompts")
        self.fs.create_file(self.baseDir / "prompt-1.prompt", contents=PROMPTS["prompt-1"])
        self.fs.create_file(self.baseDir / "prompt-2.txt", contents=PROMPTS["prompt-2"])
        self.fs.create_file(self.baseDir / "prompt-3.prompt", contents=PROMPTS["prompt-3"])
        self.fs.create_file(self.baseDir / "subPrompts" / "prompt-4.prompt", contents=PROMPTS["prompt-4"])

    def test_prompts_get(self):
        loader = PromptLoader(self.baseDir, recursive=False)
        loaded_names = loader.list_all()
        expected_names = ["prompt-1", "prompt-3"]
        assert _list_equal(loaded_names, expected_names), f"load: {loaded_names}, expect: {expected_names}"

    def test_prompts_get_recursive(self):
        assert (self.baseDir / "subPrompts").is_dir()
        loader = PromptLoader(self.baseDir, recursive=True)
        loaded_names = loader.list_all()
        expected_names = ["prompt-1", "subPrompts/prompt-4", "prompt-3"]
        assert _list_equal(loaded_names, expected_names), f"load: {loaded_names}, expect: {expected_names}"

    def test_set_suffix(self):
        loader = PromptLoader(self.baseDir, suffix=".txt", recursive=False)
        loaded_names = loader.list_all()
        expected_names = ["prompt-2"]
        assert _list_equal(loaded_names, expected_names), f"load: {loaded_names}, expect: {expected_names}"

    def tearDown(self) -> None:
        self.tearDownPyfakefs()

    def test_parse(self):
        readed_content = """
@param:param1
@param:param2
@begin
the prompt file body.
read param1: $param1
read param2: $param2
@end
""".strip()
        self.fs.create_file("/usr/prompts/prompt01.prompt", contents=readed_content)
        loader = PromptLoader(Path("/usr/prompts"))
        template = loader.load("prompt01")
        assert template is not None
        assert _list_equal(template.params, {"param1", "param2"}), f"params: {template.params}, expect: {'param1', 'param2'}"
        assert isinstance(template.template, Template)
        assert template.render(param1="value1", param2="value2") == "the prompt file body.\nread param1: value1\nread param2: value2"