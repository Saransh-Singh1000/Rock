from Logger import log_error, log_info, log_warning, ERROR_SUGGESTIONS
from Logger import TokenType, Token
from Logger import Expression, BinaryExpr, NumberExpr, StringExpr, Program, VarExpr, VarDecl, Stdout, Function, Call
from typing import List
import sys

def parse(tokens: List[Token]) -> Program:
    log_info("Starting parsing of tokens into AST")
    pos = 0
    functions = {}
    invoke = None
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
            if pos < len(tokens) and tokens[pos].type in (TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
                op_token = tokens[pos]
                pos += 1
                right = parse_expression()
                if not right:
                    return None
                return BinaryExpr(VarExpr(token.value), op_token.value, right)
            return VarExpr(token.value)
        else:
            log_error(f"Invalid expression starting with token {token.value}", token.line, ERROR_SUGGESTIONS["invalid_expression"])
            valid = False
            return None

    while pos < len(tokens):
        token = tokens[pos]
        if token.type == TokenType.DEFINE:
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
                if tokens[pos].type in (TokenType.INT, TokenType.FLOAT, TokenType.STRING):
                    var_type = tokens[pos].type.value.lower()
                    pos += 1
                    var_token = consume(TokenType.IDENTIFIER)
                    if not var_token:
                        continue
                    var_name = var_token.value
                    if var_name in variables:
                        log_error(f"Variable '{var_name}' is defined more than once in function '{func_name}'", var_token.line, ERROR_SUGGESTIONS["duplicate_variable"])
                        valid = False
                        continue
                    if not consume(TokenType.EQUALS):
                        continue
                    expr = parse_expression()
                    if not expr:
                        continue
                    # Type checking
                    if var_type == "int" and isinstance(expr, NumberExpr) and expr.type != "i64":
                        log_error(f"Type mismatch: Expected int for variable '{var_name}', got {expr.type}", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                        valid = False
                    elif var_type == "float" and isinstance(expr, NumberExpr) and expr.type != "double":
                        log_error(f"Type mismatch: Expected float for variable '{var_name}', got {expr.type}", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                        valid = False
                    elif var_type == "string" and not isinstance(expr, StringExpr):
                        log_error(f"Type mismatch: Expected string for variable '{var_name}'", var_token.line, ERROR_SUGGESTIONS["type_mismatch"])
                        valid = False
                    elif isinstance(expr, VarExpr) and expr.name not in variables:
                        log_error(f"Variable '{expr.name}' used before declaration in function '{func_name}'", var_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                        valid = False
                    elif isinstance(expr, BinaryExpr):
                        if isinstance(expr.left, VarExpr) and expr.left.name not in variables:
                            log_error(f"Variable '{expr.left.name}' used before declaration in function '{func_name}'", var_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                            valid = False
                        if isinstance(expr.right, VarExpr) and expr.right.name not in variables:
                            log_error(f"Variable '{expr.right.name}' used before declaration in function '{func_name}'", var_token.line, ERROR_SUGGESTIONS["undefined_variable"])
                            valid = False
                        if expr.op == '/' and isinstance(expr.right, NumberExpr) and expr.right.value == '0':
                            log_error(f"Division by zero detected in expression", var_token.line, ERROR_SUGGESTIONS["division_by_zero"])
                            valid = False
                    variables[var_name] = var_type
                    body.append(VarDecl(var_name, var_type, expr))
                    if not consume(TokenType.SEMICOLON):
                        continue
                elif tokens[pos].type == TokenType.STDOUT:
                    pos += 1
                    if pos < len(tokens) and tokens[pos].type == TokenType.STRING_LIT:
                        # Handle direct string literal in stdout
                        expr = StringExpr(tokens[pos].value)
                        pos += 1
                        body.append(Stdout(expr))
                        if not consume(TokenType.SEMICOLON):
                            continue
                    else:
                        expr = parse_expression()
                        if not expr:
                            continue
                        if isinstance(expr, VarExpr) and expr.name not in variables:
                            log_error(f"Variable '{expr.name}' used before declaration in function '{func_name}'", tokens[pos-1].line, ERROR_SUGGESTIONS["undefined_variable"])
                            valid = False
                        if isinstance(expr, BinaryExpr):
                            if isinstance(expr.left, VarExpr) and expr.left.name not in variables:
                                log_error(f"Variable '{expr.left.name}' used before declaration in function '{func_name}'", tokens[pos-1].line, ERROR_SUGGESTIONS["undefined_variable"])
                                valid = False
                            if isinstance(expr.right, VarExpr) and expr.right.name not in variables:
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
                invoke = invoke_token.value
                if invoke not in functions:
                    log_error(f"Attempting to invoke undefined function '{invoke}'", invoke_token.line, ERROR_SUGGESTIONS["invoking_undefined_function"])
                    valid = False
            if not consume(TokenType.SEMICOLON):
                continue
        else:
            log_error(f"Unexpected token '{token.value}'", token.line, ERROR_SUGGESTIONS["unexpected_token"])
            valid = False
            pos += 1

    if not valid:
        log_error("Parsing failed due to errors. Compilation aborted.")
        sys.exit(1)

    log_info(f"Parsing completed: {len(functions)} functions defined")
    if not functions:
        log_warning("No functions defined in the program")
    if not invoke:
        log_warning("No 'invoke' statement found. Program may not execute any function")
    return Program(functions=functions, invoke=invoke)
