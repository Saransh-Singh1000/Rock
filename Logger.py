import shutil

from rich import print
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional

# ----------------- HEADER -----------------
Size = shutil.get_terminal_size()
Text = "(C)Copyright Saransh Singh"
Padding = (Size.columns - len(Text)) // 2
print(" " * Padding + f"[blue]{Text}[/blue]")
print(f"[green]{'_' * Size.columns}[/green]")

# Compiler path
CLANG_PATH = "clang"  # Ensure clang is in PATH
GCC_PATH = "C:\\mingw64\\mingw64\\bin\\gcc.exe"  # Adjusted to match your environment

# ----------------- MESSAGE FORMATTING -----------------
def log_info(message: str) -> None:
    print(f"[cyan][INFO][/cyan] {message}")

def log_warning(message: str, line: Optional[int] = None, suggestion: Optional[str] = None) -> None:
    msg = f"[yellow][WARNING][/yellow] {message}"
    if line is not None:
        msg += f" at line {line}"
    print(msg)
    if suggestion:
        print(f"[blue][SUGGESTION][/blue] {suggestion}")

def log_error(message: str, line: Optional[int] = None, suggestion: Optional[str] = None) -> None:
    msg = f"[red][ERROR][/red] {message}"
    if line is not None:
        msg += f" at line {line}"
    print(msg)
    if suggestion:
        print(f"[blue][SUGGESTION][/blue] {suggestion}")

# ----------------- SUGGESTIONS -----------------
ERROR_SUGGESTIONS = {
    "invalid_function_declaration": "Check the 'define' syntax. Correct format: 'define FuncName { ... }'.",
    "duplicate_function": "Function names must be unique. Rename or remove the duplicate function.",
    "extra_closing_brace": "Found an unexpected '}'. Remove it or ensure a matching '{' exists.",
    "invoking_undefined_function": "Ensure the function is defined before calling it and check for spelling or case errors.",
    "duplicate_variable": "Variable names must be unique within a function. Rename or remove the duplicate variable.",
    "invalid_variable_definition": "Use correct variable syntax, e.g., 'int var = expression;'.",
    "undefined_variable": "Declare the variable before using it, e.g., 'int var = value;'.",
    "division_by_zero": "The denominator in a division cannot be zero. Add a check to avoid this.",
    "unsupported_command": "Use only valid Rock language commands (int, float, string, stdout, etc.).",
    "missing_closing_brace": "Add a closing '}' to complete the function definition.",
    "invalid_expression": "Check the expression for correct syntax and valid variable names.",
    "invalid_character": "Use only valid characters in the Rock language syntax.",
    "unterminated_string": "Ensure strings are closed with a matching quotation mark.",
    "unexpected_token": "Check the syntax at this position for misplaced or incorrect tokens.",
    "type_mismatch": "Ensure the variable type (int, float, string) matches the expression type."
}

# ----------------- TOKENIZATION -----------------
class TokenType(Enum):
    DEFINE = "DEFINE"
    IDENTIFIER = "IDENTIFIER"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    INVOKE = "INVOKE"
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    EQUALS = "EQUALS"
    NUMBER = "NUMBER"
    FLOAT_NUM = "FLOAT_NUM"
    STRING_LIT = "STRING_LIT"
    STDOUT = "STDOUT"
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MODULO = "MODULO"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    SEMICOLON = "SEMICOLON"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int


# ----------------- AST -----------------
@dataclass
class ASTNode:
    pass

@dataclass
class Program(ASTNode):
    functions: Dict[str, 'Function']
    invoke: Optional[str]

@dataclass
class Function(ASTNode):
    name: str
    body: List[ASTNode]

@dataclass
class VarDecl(ASTNode):
    name: str
    type: str  # int, float, string
    expr: 'Expression'

@dataclass
class Stdout(ASTNode):
    expr: 'Expression'

@dataclass
class Call(ASTNode):
    func_name: str

@dataclass
class Expression(ASTNode):
    pass

@dataclass
class NumberExpr(Expression):
    value: str
    type: str  # i64 or double

@dataclass
class StringExpr(Expression):
    value: str

@dataclass
class VarExpr(Expression):
    name: str

@dataclass
class BinaryExpr(Expression):
    left: Expression
    op: str
    right: Expression


