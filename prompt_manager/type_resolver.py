import ast
from typing import Any, Literal


_ALLOWED_NAMES = {
    "Any": Any,
    "str": str,
    "int": int,
    "bool": bool,
    "float": float,
    "list": list,
    "dict": dict,
    "Literal": Literal,
}

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.Name,
    ast.Subscript,
    ast.Tuple,
    ast.Constant,
    ast.Load,
    ast.BinOp,
    ast.BitOr,
)


class TypeResolver:
    class TypeHintValidator(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id not in _ALLOWED_NAMES:
                raise ValueError(
                    f"Unsupported type: {node.id}"
                )

        def visit_Call(self, node):
            raise ValueError("Function call is forbidden")

        def visit_Attribute(self, node):
            raise ValueError("Attribute access is forbidden")

        def generic_visit(self, node):
            if not isinstance(node, _ALLOWED_AST_NODES):
                raise ValueError(
                    f"Unsupported syntax: {type(node).__name__}"
                )

            super().generic_visit(node)
   
    @classmethod
    def resolve_type(cls, type_hint: str):
        ast_tree = ast.parse(type_hint, mode="eval")
        cls.TypeHintValidator().visit(ast_tree)
        return eval(
            compile(ast_tree, "<type hint>", "eval"),
            {"__builtins__": {}},
            _ALLOWED_NAMES,
        )