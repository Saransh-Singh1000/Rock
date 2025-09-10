import shutil
import os
import sys
import subprocess
import tempfile
from rich import print
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from llvmlite import ir

# ----------------- HEADER -----------------
Size = shutil.get_terminal_size()
Text = "(C)Copyright Saransh Singh"
Padding = (Size.columns - len(Text)) // 2
print(" " * Padding + f"[blue]{Text}[/blue]")
print(f"[green]{'_' * Size.columns}[/green]")

# Compiler path
CLANG_PATH = "LLVM\\LLVM\\bin\\clang.exe"  # Ensure clang is in PATH
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
    "duplicate_variable": "Variable names must be unique within a scope. Rename or remove the duplicate variable.",
    "invalid_variable_definition": "Use correct variable syntax, e.g., 'int var = expression;'.",
    "undefined_variable": "Declare the variable before using it, e.g., 'int var = value;' or 'global type var = value;'.",
    "division_by_zero": "The denominator in a division cannot be zero. Add a check to avoid this.",
    "unsupported_command": "Use only valid Rock language commands (int, float, string, stdout, global, etc.).",
    "missing_closing_brace": "Add a closing '}' to complete the function definition.",
    "invalid_expression": "Check the expression for correct syntax and valid variable names.",
    "invalid_character": "Use only valid characters in the Rock language syntax.",
    "unterminated_string": "Ensure strings are closed with a matching quotation mark.",
    "unexpected_token": "Check the syntax at this position for misplaced or incorrect tokens.",
    "type_mismatch": "Ensure the variable types are compatible or explicitly cast between int and float.",
    "invalid_global_declaration": "Global declarations must be of the form 'global type var = expr;' inside functions or at program level.",
    "duplicate_global": "Global variable names must be unique. Rename or remove the duplicate global variable.",
    "invalid_assignment": "Assignment to undefined variable. Ensure the variable is declared (locally or globally) before assignment.",
    "invalid_global_initializer": "Global variables must be initialized with a valid expression."
}

# ----------------- TOKENIZATION -----------------
class TokenType(Enum):
    DEFINE = "DEFINE"
    GLOBAL = "GLOBAL"
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
    COMMA = "COMMA"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int

def tokenize(rock_file: str) -> List[Token]:
    log_info(f"Starting tokenization of {rock_file}")
    try:
        with open(rock_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        log_error(f"Could not find file: {rock_file}")
        return []
    except Exception as e:
        log_error(f"Failed to read file: {str(e)}")
        return []

    tokens = []
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        pos = 0
        while pos < len(line):
            char = line[pos]
            if char.isspace():
                pos += 1
                continue
            if char.isalpha() or char == '_':
                identifier = ""
                while pos < len(line) and (line[pos].isalnum() or line[pos] == '_'):
                    identifier += line[pos]
                    pos += 1
                if identifier == "define":
                    tokens.append(Token(TokenType.DEFINE, identifier, lineno))
                elif identifier == "global":
                    tokens.append(Token(TokenType.GLOBAL, identifier, lineno))
                elif identifier == "invoke":
                    tokens.append(Token(TokenType.INVOKE, identifier, lineno))
                elif identifier == "int":
                    tokens.append(Token(TokenType.INT, identifier, lineno))
                elif identifier == "float":
                    tokens.append(Token(TokenType.FLOAT, identifier, lineno))
                elif identifier == "string":
                    tokens.append(Token(TokenType.STRING, identifier, lineno))
                elif identifier == "stdout":
                    tokens.append(Token(TokenType.STDOUT, identifier, lineno))
                else:
                    tokens.append(Token(TokenType.IDENTIFIER, identifier, lineno))
                continue
            if char == '"':
                string = ""
                pos += 1
                while pos < len(line) and line[pos] != '"':
                    string += line[pos]
                    pos += 1
                if pos >= len(line):
                    log_error("Unterminated string literal", lineno, ERROR_SUGGESTIONS["unterminated_string"])
                    return []
                tokens.append(Token(TokenType.STRING_LIT, string, lineno))
                pos += 1
                continue
            if char.isdigit() or (char == '-' and pos + 1 < len(line) and (line[pos + 1].isdigit() or line[pos + 1] == '.')):
                number = char
                pos += 1
                is_float = False
                while pos < len(line) and (line[pos].isdigit() or line[pos] == '.'):
                    if line[pos] == '.':
                        if is_float:
                            log_error("Invalid float literal (multiple decimals)", lineno, ERROR_SUGGESTIONS["invalid_expression"])
                            return []
                        is_float = True
                    number += line[pos]
                    pos += 1
                token_type = TokenType.FLOAT_NUM if is_float else TokenType.NUMBER
                tokens.append(Token(token_type, number, lineno))
                continue
            if char == '{':
                tokens.append(Token(TokenType.LBRACE, char, lineno))
            elif char == '}':
                tokens.append(Token(TokenType.RBRACE, char, lineno))
            elif char == '=':
                tokens.append(Token(TokenType.EQUALS, char, lineno))
            elif char == '+':
                tokens.append(Token(TokenType.PLUS, char, lineno))
            elif char == '-':
                tokens.append(Token(TokenType.MINUS, char, lineno))
            elif char == '*':
                tokens.append(Token(TokenType.MULTIPLY, char, lineno))
            elif char == '/':
                tokens.append(Token(TokenType.DIVIDE, char, lineno))
            elif char == '%':
                tokens.append(Token(TokenType.MODULO, char, lineno))
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, char, lineno))
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, char, lineno))
            elif char == ';':
                tokens.append(Token(TokenType.SEMICOLON, char, lineno))
            elif char == ',':
                tokens.append(Token(TokenType.COMMA, char, lineno))
            else:
                log_error(f"Invalid character '{char}' found", lineno, ERROR_SUGGESTIONS["invalid_character"])
                return []
            pos += 1

    log_info(f"Tokenization completed: {len(tokens)} tokens generated")
    if not tokens:
        log_warning("No valid tokens found in the file. Is the file empty?")
    return tokens

# ----------------- AST -----------------
@dataclass
class ASTNode:
    pass

@dataclass
class Program(ASTNode):
    globals: Dict[str, Tuple[str, 'Expression']]  # name -> (type, initializer)
    functions: Dict[str, 'Function']
    invoke: List[str]  # List to support multiple invokes

@dataclass
class Function(ASTNode):
    name: str
    body: List[ASTNode]

@dataclass
class GlobalVarDecl(ASTNode):
    name: str
    type: str
    expr: 'Expression'

@dataclass
class VarDecl(ASTNode):
    name: str
    type: str
    expr: 'Expression'

@dataclass
class Assignment(ASTNode):
    name: str
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
    type: str

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

def parse(tokens: List[Token]) -> Program:
    log_info("Starting parsing of tokens into AST")
    pos = 0
    functions = {}
    globals = {}
    invokes = []
    valid = True

    def consume(expected_type: TokenType) -> Token:
        nonlocal pos, valid
        if pos < len(tokens) and tokens[pos].type == expected_type:
            token = tokens[pos]
            pos += 1
            return token
        log_error(f"Expected token {expected_type.value}", tokens[pos].line if pos < len(tokens) else None, ERROR_SUGGESTIONS["unexpected_token"])
        valid = False
        return None

    def parse_expression() -> Expression:
        nonlocal pos, valid

        def parse_primary() -> Expression:
            nonlocal pos, valid
            if pos >= len(tokens):
                log_error("Unexpected end of file in expression", None, ERROR_SUGGESTIONS["invalid_expression"])
                valid = False
                return None

            token = tokens[pos]
            pos += 1

            if token.type == TokenType.NUMBER:
                return NumberExpr(token.value, "i64")
            elif token.type == TokenType.FLOAT_NUM:
                return NumberExpr(token.value, "double")
            elif token.type == TokenType.STRING_LIT:
                return StringExpr(token.value)
            elif token.type == TokenType.IDENTIFIER:
                return VarExpr(token.value)
            elif token.type == TokenType.LPAREN:
                expr = parse_additive()
                if not expr:
                    return None
                if not consume(TokenType.RPAREN):
                    return None
                return expr
            else:
                log_error(f"Invalid expression starting with token {token.value}", token.line, ERROR_SUGGESTIONS["invalid_expression"])
                valid = False
                return None

        def parse_multiplicative() -> Expression:
            nonlocal pos, valid
            expr = parse_primary()
            if not expr:
                return None
            while pos < len(tokens) and tokens[pos].type in (TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
                op_token = tokens[pos]
                pos += 1
                right = parse_primary()
                if not right:
                    return None
                expr = BinaryExpr(expr, op_token.value, right)
            return expr

        def parse_additive() -> Expression:
            nonlocal pos, valid
            expr = parse_multiplicative()
            if not expr:
                return None
            while pos < len(tokens) and tokens[pos].type in (TokenType.PLUS, TokenType.MINUS):
                op_token = tokens[pos]
                pos += 1
                right = parse_multiplicative()
                if not right:
                    return None
                expr = BinaryExpr(expr, op_token.value, right)
            return expr

        return parse_additive()

    def parse_global(is_function_scope: bool, func_name: str = None, body: List[ASTNode] = None, variables: Dict[str, str] = None) -> None:
        nonlocal pos, valid, globals
        pos += 1
        type_token = consume(TokenType.INT) or consume(TokenType.FLOAT) or consume(TokenType.STRING)
        if not type_token:
            return
        var_type = type_token.type.value.lower()
        var_token = consume(TokenType.IDENTIFIER)
        if not var_token:
            return
        var_name = var_token.value
        if var_name in globals:
            log_error(f"Global variable '{var_name}' is defined more than once", var_token.line, ERROR_SUGGESTIONS["duplicate_global"])
            valid = False
            return
        if not consume(TokenType.EQUALS):
            return
        expr = parse_expression()
        if not expr:
            return
        if isinstance(expr, NumberExpr):
            if var_type == "int" and expr.type != "i64":
                log_error(f"Type mismatch: Expected int for global variable '{var_name}', got {expr.type}", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                valid = False
            elif var_type == "float" and expr.type != "double":
                log_error(f"Type mismatch: Expected float for global variable '{var_name}', got {expr.type}", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                valid = False
            elif var_type == "string":
                log_error(f"Type mismatch: Expected string for global variable '{var_name}'", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                valid = False
        elif isinstance(expr, StringExpr) and var_type != "string":
            log_error(f"Type mismatch: Expected string for global variable '{var_name}'", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
            valid = False
        elif isinstance(expr, BinaryExpr):
            if var_type == "string":
                log_error(f"Type mismatch: Binary expressions not allowed for string global variable '{var_name}'", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                valid = False
            if isinstance(expr.left, VarExpr) and expr.left.name not in variables and expr.left.name not in globals and is_function_scope:
                log_error(f"Variable '{expr.left.name}' used before declaration in global initializer", var_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                valid = False
            if isinstance(expr.right, VarExpr) and expr.right.name not in variables and expr.right.name not in globals and is_function_scope:
                log_error(f"Variable '{expr.right.name}' used before declaration in global initializer", var_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                valid = False
            if expr.op == '/' and isinstance(expr.right, NumberExpr) and expr.right.value == '0':
                log_error(f"Division by zero detected in global initializer", var_token.line, ERROR_SUGGESTIONS["division_by_zero"])
                valid = False
        elif isinstance(expr, VarExpr) and expr.name not in variables and expr.name not in globals and is_function_scope:
            log_error(f"Variable '{expr.name}' used before declaration in global initializer", var_token.line, ERROR_SUGGESTIONS["undefined_variable"])
            valid = False
        if not consume(TokenType.SEMICOLON):
            return
        globals[var_name] = (var_type, expr)
        if is_function_scope:
            body.append(GlobalVarDecl(var_name, var_type, expr))

    while pos < len(tokens):
        token = tokens[pos]
        if token.type == TokenType.GLOBAL:
            parse_global(is_function_scope=False)
            continue
        elif token.type == TokenType.DEFINE:
            pos += 1
            name_token = consume(TokenType.IDENTIFIER)
            if not name_token:
                continue
            func_name = name_token.value
            if func_name in functions:
                log_error(f"Function '{func_name}' is defined more than once", name_token.line, ERROR_SUGGESTIONS["duplicate_function"])
                valid = False
                continue
            if not consume(TokenType.LBRACE):
                continue
            body = []
            variables = {}
            while pos < len(tokens) and tokens[pos].type != TokenType.RBRACE:
                if tokens[pos].type == TokenType.GLOBAL:
                    parse_global(is_function_scope=True, func_name=func_name, body=body, variables=variables)
                    continue
                elif tokens[pos].type in (TokenType.INT, TokenType.FLOAT, TokenType.STRING):
                    var_type = tokens[pos].type.value.lower()
                    pos += 1
                    var_token = consume(TokenType.IDENTIFIER)
                    if not var_token:
                        continue
                    var_name = var_token.value
                    if var_name in variables:
                        log_error(f"Local variable '{var_name}' is defined more than once in function '{func_name}'", var_token.line, ERROR_SUGGESTIONS["duplicate_variable"])
                        valid = False
                        continue
                    if not consume(TokenType.EQUALS):
                        continue
                    expr = parse_expression()
                    if not expr:
                        continue
                    if var_type == "int" and isinstance(expr, NumberExpr) and expr.type != "i64":
                        log_error(f"Type mismatch: Expected int for variable '{var_name}', got {expr.type}", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                        valid = False
                    elif var_type == "float" and isinstance(expr, NumberExpr) and expr.type != "double":
                        log_error(f"Type mismatch: Expected float for variable '{var_name}', got {expr.type}", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                        valid = False
                    elif var_type == "string" and not isinstance(expr, StringExpr):
                        log_error(f"Type mismatch: Expected string for variable '{var_name}'", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                        valid = False
                    elif isinstance(expr, VarExpr) and expr.name not in variables and expr.name not in globals:
                        log_error(f"Variable '{expr.name}' used before declaration in function '{func_name}'", var_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                        valid = False
                    elif isinstance(expr, BinaryExpr):
                        if isinstance(expr.left, VarExpr) and expr.left.name not in variables and expr.left.name not in globals:
                            log_error(f"Variable '{expr.left.name}' used before declaration in function '{func_name}'", var_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                            valid = False
                        if isinstance(expr.right, VarExpr) and expr.right.name not in variables and expr.right.name not in globals:
                            log_error(f"Variable '{expr.right.name}' used before declaration in function '{func_name}'", var_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                            valid = False
                        if expr.op == '/' and isinstance(expr.right, NumberExpr) and expr.right.value == '0':
                            log_error(f"Division by zero detected in expression", var_token.line, ERROR_SUGGESTIONS["division_by_zero"])
                            valid = False
                    variables[var_name] = var_type
                    body.append(VarDecl(var_name, var_type, expr))
                    if not consume(TokenType.SEMICOLON):
                        continue
                elif tokens[pos].type == TokenType.IDENTIFIER and pos + 1 < len(tokens) and tokens[pos + 1].type == TokenType.EQUALS:
                    assign_token = tokens[pos]
                    pos += 2
                    var_name = assign_token.value
                    expr = parse_expression()
                    if not expr:
                        continue
                    var_type = variables.get(var_name) or globals.get(var_name, (None, None))[0]
                    if not var_type:
                        log_error(f"Assignment to undefined variable '{var_name}' in function '{func_name}'", assign_token.line, ERROR_SUGGESTIONS["invalid_assignment"])
                        valid = False
                        continue
                    if var_type == "int" and isinstance(expr, NumberExpr) and expr.type != "i64":
                        log_error(f"Type mismatch: Expected int for assignment to '{var_name}', got {expr.type}", assign_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                        valid = False
                    elif var_type == "float" and isinstance(expr, NumberExpr) and expr.type != "double":
                        log_error(f"Type mismatch: Expected float for assignment to '{var_name}', got {expr.type}", assign_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                        valid = False
                    elif var_type == "string" and not isinstance(expr, StringExpr):
                        log_error(f"Type mismatch: Expected string for assignment to '{var_name}'", assign_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                        valid = False
                    elif isinstance(expr, VarExpr) and expr.name not in variables and expr.name not in globals:
                        log_error(f"Variable '{expr.name}' used before declaration in function '{func_name}'", assign_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                        valid = False
                    elif isinstance(expr, BinaryExpr):
                        if isinstance(expr.left, VarExpr) and expr.left.name not in variables and expr.left.name not in globals:
                            log_error(f"Variable '{expr.left.name}' used before declaration in function '{func_name}'", assign_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                            valid = False
                        if isinstance(expr.right, VarExpr) and expr.right.name not in variables and expr.right.name not in globals:
                            log_error(f"Variable '{expr.right.name}' used before declaration in function '{func_name}'", assign_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                            valid = False
                        if expr.op == '/' and isinstance(expr.right, NumberExpr) and expr.right.value == '0':
                            log_error(f"Division by zero detected in expression", assign_token.line, ERROR_SUGGESTIONS["division_by_zero"])
                            valid = False
                    body.append(Assignment(var_name, expr))
                    if not consume(TokenType.SEMICOLON):
                        continue
                elif tokens[pos].type == TokenType.STDOUT:
                    pos += 1
                    if pos < len(tokens) and tokens[pos].type == TokenType.STRING_LIT:
                        expr = StringExpr(tokens[pos].value)
                        pos += 1
                        body.append(Stdout(expr))
                        if not consume(TokenType.SEMICOLON):
                            continue
                    else:
                        expr = parse_expression()
                        if not expr:
                            continue
                        if isinstance(expr, VarExpr) and expr.name not in variables and expr.name not in globals:
                            log_error(f"Variable '{expr.name}' used before declaration in function '{func_name}'", tokens[pos-1].line, ERROR_SUGGESTIONS["undefined_variable"])
                            valid = False
                        if isinstance(expr, BinaryExpr):
                            if isinstance(expr.left, VarExpr) and expr.left.name not in variables and expr.left.name not in globals:
                                log_error(f"Variable '{expr.left.name}' used before declaration in function '{func_name}'", tokens[pos-1].line, ERROR_SUGGESTIONS["undefined_variable"])
                                valid = False
                            if isinstance(expr.right, VarExpr) and expr.right.name not in variables and expr.right.name not in globals:
                                log_error(f"Variable '{expr.right.name}' used before declaration in function '{func_name}'", tokens[pos-1].line, ERROR_SUGGESTIONS["undefined_variable"])
                                valid = False
                            if expr.op == '/' and isinstance(expr.right, NumberExpr) and expr.right.value == '0':
                                log_error(f"Division by zero detected in expression", tokens[pos-1].line, ERROR_SUGGESTIONS["division_by_zero"])
                                valid = False
                        body.append(Stdout(expr))
                        if not consume(TokenType.SEMICOLON):
                            continue
                elif tokens[pos].type == TokenType.IDENTIFIER and pos + 1 < len(tokens) and tokens[pos + 1].type == TokenType.LPAREN:
                    func_name_call = tokens[pos].value
                    pos += 2
                    if not consume(TokenType.RPAREN):
                        continue
                    if func_name_call not in functions and func_name_call != func_name:
                        log_error(f"Calling undefined function '{func_name_call}' in function '{func_name}'", tokens[pos-2].line, ERROR_SUGGESTIONS["invoking_undefined_function"])
                        valid = False
                    body.append(Call(func_name_call))
                    if not consume(TokenType.SEMICOLON):
                        continue
                else:
                    log_error(f"Unsupported command or syntax error", tokens[pos].line, ERROR_SUGGESTIONS["unsupported_command"])
                    valid = False
                    pos += 1
            if not consume(TokenType.RBRACE):
                log_error(f"Missing closing brace '}}' for function '{func_name}'", tokens[pos-1].line if pos > 0 else None, ERROR_SUGGESTIONS["missing_closing_brace"])
                valid = False
                continue
            functions[func_name] = Function(func_name, body)
        elif token.type == TokenType.INVOKE:
            pos += 1
            invoke_token = consume(TokenType.IDENTIFIER)
            if invoke_token:
                invoke_name = invoke_token.value
                if invoke_name not in functions:
                    log_error(f"Attempting to invoke undefined function '{invoke_name}'", invoke_token.line, ERROR_SUGGESTIONS["invoking_undefined_function"])
                    valid = False
                else:
                    invokes.append(invoke_name)
            if not consume(TokenType.SEMICOLON):
                continue
        else:
            log_error(f"Unexpected token '{token.value}'", token.line, ERROR_SUGGESTIONS["unexpected_token"])
            valid = False
            pos += 1

    if not valid:
        log_error("Parsing failed due to errors. Compilation aborted.")
        sys.exit(1)

    log_info(f"Parsing completed: {len(functions)} functions and {len(globals)} global variables defined")
    if not functions:
        log_warning("No functions defined in the program")
    if not invokes:
        log_warning("No 'invoke' statement found. Program may not execute any function")
    return Program(globals=globals, functions=functions, invoke=invokes)

# ----------------- LLVM IR GENERATION -----------------
@dataclass
class LLVMContext:
    var_map: Dict[str, Dict[str, Tuple[ir.Type, ir.AllocaInstr]]]  # func -> var -> (type, alloca)
    global_map: Dict[str, Tuple[ir.Type, ir.GlobalVariable]]  # global var -> (type, global)
    string_map: Dict[Tuple[str, str], ir.GlobalVariable]  # (func, string) -> global string
    reg_count: int
    builder: Optional[ir.IRBuilder]

def generate_llvm_ir(program: Program, ll_file: str):
    log_info(f"Generating LLVM IR to {ll_file}")
    module = ir.Module(name=ll_file)
    context = LLVMContext(var_map={}, global_map={}, string_map={}, reg_count=0, builder=None)

    def new_reg() -> str:
        context.reg_count += 1
        return f"%{context.reg_count}"

    def map_rock_to_llvm_type(rock_type: str) -> ir.Type:
        return {"int": ir.IntType(64), "float": ir.DoubleType(), "string": ir.PointerType(ir.IntType(8))}[rock_type]

    # Declare printf
    i32 = ir.IntType(32)
    i8_ptr = ir.PointerType(ir.IntType(8))
    printf_type = ir.FunctionType(i32, [i8_ptr], var_arg=True)
    printf = ir.Function(module, printf_type, name="printf")

    # String format constants
    str_int = ir.GlobalVariable(module, ir.ArrayType(ir.IntType(8), 5), name=".str.int")
    str_int.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 5), bytearray("%ld\n\0", "utf-8"))
    str_int.linkage = "private"
    str_int.global_constant = True

    str_double = ir.GlobalVariable(module, ir.ArrayType(ir.IntType(8), 4), name=".str.double")
    str_double.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 4), bytearray("%f\n\0", "utf-8"))
    str_double.linkage = "private"
    str_double.global_constant = True

    str_no_newline = ir.GlobalVariable(module, ir.ArrayType(ir.IntType(8), 3), name=".str.no_newline")
    str_no_newline.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 3), bytearray("%s\0", "utf-8"))
    str_no_newline.linkage = "private"
    str_no_newline.global_constant = True

    def process_expr(expr: Expression, func_name: str, builder: ir.IRBuilder) -> Tuple[ir.Value, ir.Type]:
        if isinstance(expr, NumberExpr):
            if expr.type == "i64":
                return ir.Constant(ir.IntType(64), int(expr.value)), ir.IntType(64)
            elif expr.type == "double":
                return ir.Constant(ir.DoubleType(), float(expr.value)), ir.DoubleType()
        elif isinstance(expr, StringExpr):
            key = (func_name, expr.value)
            if key not in context.string_map:
                escaped_str = expr.value + "\0"
                str_type = ir.ArrayType(ir.IntType(8), len(escaped_str))
                str_global = ir.GlobalVariable(module, str_type, name=f".str.{len(context.string_map) + 1}")
                str_global.initializer = ir.Constant(str_type, bytearray(escaped_str, "utf-8"))
                str_global.linkage = "private"
                str_global.global_constant = True
                context.string_map[key] = str_global
            str_global = context.string_map[key]
            ptr = builder.gep(str_global, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
            return ptr, i8_ptr
        elif isinstance(expr, VarExpr):
            if expr.name in context.var_map.get(func_name, {}):
                var_type, var_alloca = context.var_map[func_name][expr.name]
                value = builder.load(var_alloca)
                return value, var_type
            elif expr.name in context.global_map:
                var_type, var_global = context.global_map[expr.name]
                value = builder.load(var_global)
                return value, var_type
            else:
                log_error(f"Variable '{expr.name}' used but not defined in function '{func_name}' or as global", None, ERROR_SUGGESTIONS["undefined_variable"])
                sys.exit(1)
        elif isinstance(expr, BinaryExpr):
            left_val, left_type = process_expr(expr.left, func_name, builder)
            right_val, right_type = process_expr(expr.right, func_name, builder)
            if expr.op == '/' and isinstance(expr.right, NumberExpr) and expr.right.value == '0':
                log_error(f"Division by zero in expression", None, ERROR_SUGGESTIONS["division_by_zero"])
                sys.exit(1)
            
            # Handle type mismatch by casting to double if one operand is double
            result_type = left_type
            if left_type != right_type:
                if isinstance(left_type, ir.IntType) and isinstance(right_type, ir.DoubleType):
                    left_val = builder.sitofp(left_val, ir.DoubleType(), name=new_reg())
                    result_type = ir.DoubleType()
                elif isinstance(left_type, ir.DoubleType) and isinstance(right_type, ir.IntType):
                    right_val = builder.sitofp(right_val, ir.DoubleType(), name=new_reg())
                    result_type = ir.DoubleType()
                else:
                    log_error(f"Binary operation not supported between types '{left_type}' and '{right_type}'", None, ERROR_SUGGESTIONS["type_mismatch"])
                    sys.exit(1)
            
            if isinstance(result_type, ir.IntType):
                op_map = {'+': builder.add, '-': builder.sub, '*': builder.mul, '/': builder.sdiv, '%': builder.srem}
                return op_map[expr.op](left_val, right_val, name=new_reg()), result_type
            elif isinstance(result_type, ir.DoubleType):
                op_map = {'+': builder.fadd, '-': builder.fsub, '*': builder.fmul, '/': builder.fdiv, '%': builder.frem}
                return op_map[expr.op](left_val, right_val, name=new_reg()), result_type
            else:
                log_error(f"Binary operation not supported for type '{result_type}'", None, ERROR_SUGGESTIONS["type_mismatch"])
                sys.exit(1)
        log_error(f"Invalid expression in function '{func_name}'", None, ERROR_SUGGESTIONS["invalid_expression"])
        sys.exit(1)

    try:
        # Declare globals (without initializers that depend on local variables)
        for var_name, (var_type, expr) in program.globals.items():
            llvm_type = map_rock_to_llvm_type(var_type)
            global_var = ir.GlobalVariable(module, llvm_type, name=var_name)
            if isinstance(expr, NumberExpr):
                if var_type == "int":
                    global_var.initializer = ir.Constant(ir.IntType(64), int(expr.value))
                elif var_type == "float":
                    global_var.initializer = ir.Constant(ir.DoubleType(), float(expr.value))
            elif isinstance(expr, StringExpr):
                key = ("global", expr.value)
                if key not in context.string_map:
                    escaped_str = expr.value + "\0"
                    str_type = ir.ArrayType(ir.IntType(8), len(escaped_str))
                    str_global = ir.GlobalVariable(module, str_type, name=f".str.{len(context.string_map) + 1}")
                    str_global.initializer = ir.Constant(str_type, bytearray(escaped_str, "utf-8"))
                    str_global.linkage = "private"
                    str_global.global_constant = True
                    context.string_map[key] = str_global
                global_var.initializer = ir.Constant(ir.PointerType(ir.IntType(8)), context.string_map[key])
            else:
                global_var.initializer = ir.Constant(llvm_type, 0) if var_type != "string" else ir.Constant(i8_ptr, None)
            context.global_map[var_name] = (llvm_type, global_var)

        # Generate functions
        for func_name, func in program.functions.items():
            func_type = ir.FunctionType(ir.VoidType(), [])
            llvm_func = ir.Function(module, func_type, name=func_name)
            entry_block = llvm_func.append_basic_block("entry")
            builder = ir.IRBuilder(entry_block)
            context.var_map[func_name] = {}
            context.builder = builder

            for stmt in func.body:
                if isinstance(stmt, GlobalVarDecl):
                    expr_val, expr_type = process_expr(stmt.expr, func_name, builder)
                    if expr_type != map_rock_to_llvm_type(stmt.type):
                        if isinstance(expr_type, ir.IntType) and stmt.type == "float":
                            expr_val = builder.sitofp(expr_val, ir.DoubleType(), name=new_reg())
                        elif isinstance(expr_type, ir.DoubleType) and stmt.type == "int":
                            expr_val = builder.fptosi(expr_val, ir.IntType(64), name=new_reg())
                        else:
                            log_error(f"Type mismatch in global variable declaration '{stmt.name}'", None, ERROR_SUGGESTIONS["type_mismatch"])
                            sys.exit(1)
                    var_type, var_global = context.global_map[stmt.name]
                    builder.store(expr_val, var_global)
                elif isinstance(stmt, VarDecl):
                    llvm_type = map_rock_to_llvm_type(stmt.type)
                    var_alloca = builder.alloca(llvm_type, name=stmt.name)
                    context.var_map[func_name][stmt.name] = (llvm_type, var_alloca)
                    expr_val, expr_type = process_expr(stmt.expr, func_name, builder)
                    if expr_type != llvm_type:
                        if isinstance(expr_type, ir.IntType) and stmt.type == "float":
                            expr_val = builder.sitofp(expr_val, ir.DoubleType(), name=new_reg())
                        elif isinstance(expr_type, ir.DoubleType) and stmt.type == "int":
                            expr_val = builder.fptosi(expr_val, ir.IntType(64), name=new_reg())
                        else:
                            log_error(f"Type mismatch in variable declaration '{stmt.name}'", None, ERROR_SUGGESTIONS["type_mismatch"])
                            sys.exit(1)
                    builder.store(expr_val, var_alloca)
                elif isinstance(stmt, Assignment):
                    expr_val, expr_type = process_expr(stmt.expr, func_name, builder)
                    if stmt.name in context.var_map[func_name]:
                        var_type, var_alloca = context.var_map[func_name][stmt.name]
                        if expr_type != var_type:
                            if isinstance(expr_type, ir.IntType) and var_type == ir.DoubleType():
                                expr_val = builder.sitofp(expr_val, ir.DoubleType(), name=new_reg())
                            elif isinstance(expr_type, ir.DoubleType) and var_type == ir.IntType(64):
                                expr_val = builder.fptosi(expr_val, ir.IntType(64), name=new_reg())
                            else:
                                log_error(f"Type mismatch in assignment to local '{stmt.name}'", None, ERROR_SUGGESTIONS["type_mismatch"])
                                sys.exit(1)
                        builder.store(expr_val, var_alloca)
                    elif stmt.name in context.global_map:
                        var_type, var_global = context.global_map[stmt.name]
                        if expr_type != var_type:
                            if isinstance(expr_type, ir.IntType) and var_type == ir.DoubleType():
                                expr_val = builder.sitofp(expr_val, ir.DoubleType(), name=new_reg())
                            elif isinstance(expr_type, ir.DoubleType) and var_type == ir.IntType(64):
                                expr_val = builder.fptosi(expr_val, ir.IntType(64), name=new_reg())
                            else:
                                log_error(f"Type mismatch in assignment to global '{stmt.name}'", None, ERROR_SUGGESTIONS["type_mismatch"])
                                sys.exit(1)
                        builder.store(expr_val, var_global)
                    else:
                        log_error(f"Assignment to undefined variable '{stmt.name}'", None, ERROR_SUGGESTIONS["invalid_assignment"])
                        sys.exit(1)
                elif isinstance(stmt, Stdout):
                    expr_val, expr_type = process_expr(stmt.expr, func_name, builder)
                    if isinstance(expr_type, ir.IntType):
                        fmt_ptr = builder.gep(str_int, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
                        builder.call(printf, [fmt_ptr, expr_val])
                    elif isinstance(expr_type, ir.DoubleType):
                        fmt_ptr = builder.gep(str_double, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
                        builder.call(printf, [fmt_ptr, expr_val])
                    elif isinstance(expr_type, ir.PointerType):
                        fmt_ptr = builder.gep(str_no_newline, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
                        builder.call(printf, [fmt_ptr, expr_val])
                    else:
                        log_error(f"Unsupported type '{expr_type}' for stdout", None, ERROR_SUGGESTIONS["type_mismatch"])
                        sys.exit(1)
                elif isinstance(stmt, Call):
                    if stmt.func_name not in program.functions:
                        log_error(f"Calling undefined function '{stmt.func_name}'", None, ERROR_SUGGESTIONS["invoking_undefined_function"])
                        sys.exit(1)
                    called_func = module.get_global(stmt.func_name)
                    builder.call(called_func, [])
            builder.ret_void()

        # Main function
        main_type = ir.FunctionType(ir.IntType(32), [])
        main_func = ir.Function(module, main_type, name="main")
        main_block = main_func.append_basic_block("entry")
        builder = ir.IRBuilder(main_block)
        context.var_map["main"] = {}

        for invoke_name in program.invoke:
            if invoke_name in program.functions:
                called_func = module.get_global(invoke_name)
                builder.call(called_func, [])
            else:
                log_error(f"Invoking undefined function '{invoke_name}'", None, ERROR_SUGGESTIONS["invoking_undefined_function"])
                sys.exit(1)

        builder.ret(ir.Constant(ir.IntType(32), 0))

        # Write LLVM IR to file
        with open(ll_file, "w", encoding="utf-8") as f:
            f.write(str(module))
        log_info("LLVM IR generation completed")
    except Exception as e:
        log_error(f"Failed to generate LLVM IR: {str(e)}")
        sys.exit(1)

# ----------------- COMPILE LLVM TO EXE -----------------
def compile_llvm_to_exe(ll_file: str, exe_name: str):
    log_info(f"Compiling LLVM IR to executable: {exe_name}")
    exe_file = os.path.join(os.getcwd(), exe_name)
    obj_file = os.path.splitext(exe_file)[0] + ".o"
    try:
        result = subprocess.run([CLANG_PATH, "-c", ll_file, "-o", obj_file], capture_output=True, text=True, check=True)
        log_info(f"Object file created: {obj_file}")
        result = subprocess.run([GCC_PATH, obj_file, "-o", exe_file], capture_output=True, text=True, check=True)
        log_info(f"Executable created successfully: {exe_file}")
    except subprocess.CalledProcessError as e:
        log_error(f"Compilation failed: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        log_error(f"Clang compiler not found at {CLANG_PATH}. Ensure Clang is installed and in PATH.")
        sys.exit(1)
    finally:
        if os.path.exists(ll_file):
            try:
                os.remove(ll_file)
                log_info(f"Temporary LLVM IR file removed: {ll_file}")
            except Exception as e:
                log_warning(f"Could not remove temporary LLVM IR file {ll_file}: {str(e)}")
        if os.path.exists(obj_file):
            try:
                os.remove(obj_file)
                log_info(f"Temporary object file removed: {obj_file}")
            except Exception as e:
                log_warning(f"Could not remove temporary object file {obj_file}: {str(e)}")

# ----------------- COMPILATION -----------------
def rock_to_llvm(rock_file: str) -> str:
    log_info(f"Starting compilation of {rock_file} to LLVM IR")
    temp_ll = tempfile.NamedTemporaryFile(delete=False, suffix=".ll")
    ll_file = temp_ll.name
    temp_ll.close()

    tokens = tokenize(rock_file)
    if not tokens:
        log_error("Tokenization failed. Compilation aborted.")
        sys.exit(1)

    program = parse(tokens)
    
    generate_llvm_ir(program, ll_file)
    
    log_info(f"LLVM IR file generated: {ll_file}")
    return ll_file

# ----------------- TERMINAL -----------------
def Terminal():
    log_info("Starting Rock compiler terminal")
    if not shutil.which(CLANG_PATH):
        log_error(f"Clang compiler not found at {CLANG_PATH}. Ensure Clang is installed and in PATH.")
        sys.exit(1)
    if not os.path.exists(GCC_PATH):
        log_error(f"GCC compiler not found at {GCC_PATH}. Ensure GCC is installed.")
        sys.exit(1)
    while True:
        print(f"[yellow]{os.getcwd()}[/yellow]: ", end="")
        inp = input().strip()
        if not inp:
            continue
        tokens = inp.split()
        cmd = tokens[0].lower()
        if cmd == "rock" and len(tokens) >= 2:
            rock_file = tokens[1]
            if not os.path.exists(rock_file):
                log_error(f"File not found: {rock_file}")
                continue
            ll_file = rock_to_llvm(rock_file)
            exe_name = os.path.splitext(os.path.basename(rock_file))[0] + ".exe"
            compile_llvm_to_exe(ll_file, exe_name)
        elif cmd == "cd":
            try:
                os.chdir(" ".join(tokens[1:]))
                log_info(f"Changed directory to {os.getcwd()}")
            except FileNotFoundError:
                log_error(f"Directory not found: {' '.join(tokens[1:])}")
            except Exception as e:
                log_error(f"Failed to change directory: {str(e)}")
        elif cmd == "exit":
            log_info("Exiting terminal")
            break
        else:
            log_error(f"Unknown command: {tokens[0]}")
            log_info("Supported commands: 'rock <filename>', 'cd <directory>', 'exit'")

# ----------------- MAIN -----------------
try:
    Terminal()
except KeyboardInterrupt:
    log_info("Terminal interrupted by user. Exiting...")
    sys.exit(0)
except Exception as e:
    log_error(f"Unexpected error in terminal: {str(e)}")
    sys.exit(1)