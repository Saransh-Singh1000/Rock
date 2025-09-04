from rich import print
from Logger import log_info, log_error, log_warning, ERROR_SUGGESTIONS
from Logger import TokenType, Token
from typing import List

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
            else:
                log_error(f"Invalid character '{char}' found", lineno, ERROR_SUGGESTIONS["invalid_character"])
                return []
            pos += 1

    log_info(f"Tokenization completed: {len(tokens)} tokens generated")
    if not tokens:
        log_warning("No valid tokens found in the file. Is the file empty?")
    return tokens
