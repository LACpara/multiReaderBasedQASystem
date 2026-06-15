import unittest
from typing import Any, Literal
from prompt_manager.type_resolver import TypeResolver

class TestTypeResolver(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.test_cases_simple = [
            ("Any", Any),
            ("str", str),
            ("int", int),
            ("bool", bool),
            ("float", float),
            ("list", list),
            ("dict", dict),
            ('Literal["1", "2", "3"]', Literal["1", "2", "3"]),
        ]

        self.test_cases_union = [
            ("int | str", int | str),
            ("list[int] | dict[str, Any]", list[int] | dict[str, Any]),
            ('Literal["1", "2", "3"] | int', Literal["1", "2", "3"] | int),
            ('Literal["1", "2", "3"] | str', Literal["1", "2", "3"] | str),
            ('Literal["1", "2", "3"] | bool', Literal["1", "2", "3"] | bool),
            ('Literal["1", "2", "3"] | float', Literal["1", "2", "3"] | float),
            ('Literal["1", "2", "3"] | list', Literal["1", "2", "3"] | list),
            ('Literal["1", "2", "3"] | dict', Literal["1", "2", "3"] | dict),
            ('Literal["1", "2", "3"] | Any', Literal["1", "2", "3"] | Any),
        ]

        self.test_cases_complex = [
            ('Literal["1", "2", "3"] | int | str', Literal["1", "2", "3"] | int | str),
            ('Literal["1", "2", "3"] | int | str | bool', Literal["1", "2", "3"] | int | str | bool),
            ('Literal["1", "2", "3"] | int | str | bool | float', Literal["1", "2", "3"] | int | str | bool | float),
            ("list[int | str | dict[str | int, str | list[Literal[\"1\", \"2\", \"3\"]]]]", list[int | str | dict[str | int, str | list[Literal["1", "2", "3"]]]]),
        ]

        self.failed_types = [
            "tuple",
            "tuple[int]",
            "tuple[int, str, bool]",
            "set",
            "set[int]",
            "set[str]",
            "set[bool]",
            "set[float]",
            "set[dict[str, Any]]",
            "set[Literal[1, 2, 3]]",
            "set[Literal[1, 2, \"3\"]]",
            "set[Literal[1, \"2\", \"3\"]]",
            "set[Literal[1, \"2\", \"3\"]]",
            "set[Literal[1, \"2\", \"3\"]]",
            "set[Literal[1, \"2\", \"3\"]]",
            "set[Literal[1, \"2\", \"3\"]]",
            "set[Literal[1, \"2\", \"3\"]]"
        ]

    def test_type_resolution_simple(self):
        for type_hint, expected in self.test_cases_simple:
            self.assertEqual(TypeResolver.resolve_type(type_hint), expected)
    
    def test_type_resolution_union(self):
        for type_hint, expected in self.test_cases_union:
            self.assertEqual(TypeResolver.resolve_type(type_hint), expected)
    
    def test_type_resolution_complex(self):
        for type_hint, expected in self.test_cases_complex:
            self.assertEqual(TypeResolver.resolve_type(type_hint), expected)

    def test_type_resolution_complex_invalid(self):
        for type_hint in self.failed_types:
            with self.subTest(type_hint=type_hint):
                with self.assertRaises(ValueError):
                    TypeResolver.resolve_type(type_hint)
    
if __name__ == "__main__":
    unittest.main()