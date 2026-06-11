"""
SimpLang Code Generator — Walks the AST and produces Python code.
Also handles module imports (modules beyond stdlib).
"""

import os
import sys
from ast_nodes import *
from lexer import Lexer
from parser import Parser


class CodeGenError(Exception):
    """Error during code generation."""
    pass


class CodeGenerator:
    """Converts AST into Python source code."""

    def __init__(self):
        self.indent_level = 0
        self.indent_size = 4
        self.lines = []
        # Track function names for scope analysis
        self.function_names = set()
        # Module system
        self.loaded_modules = {}

    def indent(self):
        return " " * (self.indent_level * self.indent_size)

    def emit(self, code: str = ""):
        if code:
            self.lines.append(self.indent() + code)
        else:
            self.lines.append("")

    def generate(self, program: Program) -> str:
        """Generate Python code from a Program AST node."""
        self.lines = []
        self.indent_level = 0
        self.function_names = set()

        for stmt in program.statements:
            self.gen_statement(stmt)

        return "\n".join(self.lines)

    def gen_statement(self, stmt):
        """Generate code for a single statement."""
        if isinstance(stmt, SayStmt):
            self.emit(f"print({self.gen_expression(stmt.expr)})")

        elif isinstance(stmt, AskStmt):
            if stmt.expr:
                self.emit(f"SimpLang_input({self.gen_expression(stmt.expr)})")
            else:
                self.emit("SimpLang_input()")

        elif isinstance(stmt, AssignStmt):
            self.emit(f"{stmt.name} = {self.gen_expression(stmt.expr)}")

        elif isinstance(stmt, IfStmt):
            self.emit(f"if {self.gen_expression(stmt.condition)}:")
            self.indent_level += 1
            for s in stmt.body:
                self.gen_statement(s)
            self.indent_level -= 1
            if stmt.else_body:
                self.emit("else:")
                self.indent_level += 1
                for s in stmt.else_body:
                    self.gen_statement(s)
                self.indent_level -= 1

        elif isinstance(stmt, WhileLoop):
            self.emit(f"while {self.gen_expression(stmt.condition)}:")
            self.indent_level += 1
            for s in stmt.body:
                self.gen_statement(s)
            self.indent_level -= 1

        elif isinstance(stmt, ForRangeLoop):
            self.emit(
                f"for {stmt.var_name} in range({self.gen_expression(stmt.start)}, "
                f"({self.gen_expression(stmt.end)}) + 1):"
            )
            self.indent_level += 1
            for s in stmt.body:
                self.gen_statement(s)
            self.indent_level -= 1

        elif isinstance(stmt, ForInLoop):
            self.emit(
                f"for {stmt.var_name} in {self.gen_expression(stmt.iterable)}:"
            )
            self.indent_level += 1
            for s in stmt.body:
                self.gen_statement(s)
            self.indent_level -= 1

        elif isinstance(stmt, FunDef):
            self.function_names.add(stmt.name)
            params = ", ".join(stmt.params)
            self.emit(f"def {stmt.name}({params}):")
            self.indent_level += 1
            for s in stmt.body:
                self.gen_statement(s)
            self.indent_level -= 1

        elif isinstance(stmt, ReturnStmt):
            self.emit(f"return {self.gen_expression(stmt.expr)}")

        elif isinstance(stmt, ImportStmt):
            # Handle module imports
            if stmt.name == "stdlib":
                self.emit("import sys, os")
                self.emit("_simp_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src') if '__file__' in dir() else 'src'")
                self.emit("sys.path.insert(0, _simp_src)")
                self.emit("import stdlib as __simp_stdlib__")
            else:
                # Custom module import
                self.emit(f"import {stmt.name}")
            self.loaded_modules[stmt.name] = True

        elif isinstance(stmt, ExprStmt):
            expr_code = self.gen_expression(stmt.expr)
            if expr_code:
                self.emit(expr_code)

    def gen_expression(self, expr) -> str:
        """Generate Python code for an expression AST node."""
        if isinstance(expr, NumberLiteral):
            return str(expr.value)

        if isinstance(expr, StringLiteral):
            # Escape for Python
            val = expr.value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'"{val}"'

        if isinstance(expr, BoolLiteral):
            return "True" if expr.value else "False"

        if isinstance(expr, NoneLiteral):
            return "None"

        if isinstance(expr, Identifier):
            return expr.name

        if isinstance(expr, BinaryOp):
            left = self.gen_expression(expr.left)
            right = self.gen_expression(expr.right)
            if expr.op == '=':
                return f"{left} = {right}"
            return f"({left} {expr.op} {right})"

        if isinstance(expr, UnaryOp):
            right = self.gen_expression(expr.right)
            return f"({expr.op} {right})"

        if isinstance(expr, Call):
            func = self.gen_expression(expr.func)
            args = ", ".join(self.gen_expression(a) for a in expr.args)
            return f"{func}({args})"

        if isinstance(expr, ListLiteral):
            items = ", ".join(self.gen_expression(i) for i in expr.items)
            return f"[{items}]"

        if isinstance(expr, Subscript):
            obj = self.gen_expression(expr.expr)
            index = self.gen_expression(expr.index)
            return f"{obj}[{index}]"

        return ""


def compile_simp_to_py(source_code: str, source_file: str = "<unknown>") -> str:
    """Full pipeline: Lex → Parse → CodeGen → Python code."""
    # Lex
    lexer = Lexer(source_code, source_file)
    tokens = lexer.tokenize()

    # Parse
    parser = Parser(tokens, source_file)
    program = parser.parse()

    # Code generation
    gen = CodeGenerator()
    python_code = gen.generate(program)

    # Add bootstrap imports for built-in functions
    bootstrap = (
        "# SimpLang compiled output\n"
        "# (Auto-generated — do not edit directly)\n"
        "import sys, os\n"
        "_simp_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'\n"
        "_simp_src = os.path.join(_simp_dir, '..', 'src')\n"
        "sys.path.insert(0, _simp_src)\n"
        "import src.stdlib as __simp_stdlib__\n"
        "SimpLang_input = __simp_stdlib__.simp_input\n"
        "convert_to_txt = __simp_stdlib__.convert_to_txt\n"
        "convert_to_num = __simp_stdlib__.convert_to_num\n"
        "convert_to_decimal = __simp_stdlib__.convert_to_decimal\n"
        "convert_to_list = __simp_stdlib__.convert_to_list\n"
        "get_type = __simp_stdlib__.get_type\n"
        "type_of = __simp_stdlib__.get_type\n"
        "\n"
    )

    return bootstrap + python_code