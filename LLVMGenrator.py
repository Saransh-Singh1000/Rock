from Logger import log_error,log_info, ERROR_SUGGESTIONS
from Logger import Expression, BinaryExpr, NumberExpr, StringExpr, Program, VarExpr, VarDecl, Stdout, Function, Call
from Parser import Program
from dataclasses import dataclass
from typing import Dict, Tuple, List
import sys

# ----------------- LLVM IR GENERATION -----------------
@dataclass
class LLVMContext:
    var_map: Dict[str, Dict[str, Tuple[str, str]]]  # func -> var -> (llvm_type, reg)
    string_map: Dict[Tuple[str, str], str]  # (func, string) -> label
    reg_count: int
    label_count: int
    string_decls: List[str]  # Store string constant declarations

def generate_llvm_ir(program: Program, ll_file: str):
    log_info(f"Generating LLVM IR to {ll_file}")
    context = LLVMContext(var_map={}, string_map={}, reg_count=0, label_count=0, string_decls=[])

    def new_reg() -> str:
        context.reg_count += 1
        return f"%{context.reg_count}"

    def new_label() -> str:
        context.label_count += 1
        return f"@.str.{context.label_count}"

    def map_rock_to_llvm_type(rock_type: str) -> str:
        return {"int": "i64", "float": "double", "string": "i8*"}[rock_type]

    def process_expr(expr: Expression, func_name: str, f) -> Tuple[str, str]:
        if isinstance(expr, NumberExpr):
            return expr.value, expr.type
        elif isinstance(expr, StringExpr):
            key = (func_name, expr.value)
            if key not in context.string_map:
                label = new_label()
                context.string_map[key] = label
                escaped_str = expr.value.replace("\n", "\\0A").replace('"', '\\22')
                context.string_decls.append(f'{label} = private constant [{len(expr.value) + 1} x i8] c"{escaped_str}\\00"')
            return context.string_map[key], "i8*"
        elif isinstance(expr, VarExpr):
            if expr.name not in context.var_map[func_name]:
                log_error(f"Variable '{expr.name}' used but not defined in function '{func_name}'", None, ERROR_SUGGESTIONS["undefined_variable"])
                sys.exit(1)
            llvm_type, var_reg = context.var_map[func_name][expr.name]
            reg = new_reg()
            f.write(f'  {reg} = load {llvm_type}, {llvm_type}* {var_reg}\n')
            return reg, llvm_type
        elif isinstance(expr, BinaryExpr):
            left_reg, left_type = process_expr(expr.left, func_name, f)
            right_reg, right_type = process_expr(expr.right, func_name, f)
            if left_type != right_type:
                log_error(f"Type mismatch in binary operation: {left_type} vs {right_type}", None, ERROR_SUGGESTIONS["type_mismatch"])
                sys.exit(1)
            if expr.op == '/' and isinstance(expr.right, NumberExpr) and expr.right.value == '0':
                log_error(f"Division by zero in expression", None, ERROR_SUGGESTIONS["division_by_zero"])
                sys.exit(1)
            reg = new_reg()
            if left_type == "i64":
                op = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'sdiv', '%': 'srem'}[expr.op]
                f.write(f'  {reg} = {op} i64 {left_reg}, {right_reg}\n')
            elif left_type == "double":
                op = {'+': 'fadd', '-': 'fsub', '*': 'fmul', '/': 'fdiv', '%': 'frem'}[expr.op]
                f.write(f'  {reg} = {op} double {left_reg}, {right_reg}\n')
            else:
                log_error(f"Binary operation not supported for type '{left_type}'", None, ERROR_SUGGESTIONS["type_mismatch"])
                sys.exit(1)
            return reg, left_type
        log_error(f"Invalid expression in function '{func_name}'", None, ERROR_SUGGESTIONS["invalid_expression"])
        sys.exit(1)

    try:
        with open(ll_file, "w", encoding="utf-8") as f:
            # Declare printf and standard format strings
            f.write('declare i32 @printf(i8*, ...)\n')
            f.write('@.str.int = private constant [5 x i8] c"%ld\\0A\\00"\n')
            f.write('@.str.double = private constant [4 x i8] c"%f\\0A\\00"\n')

            # Emit all string constants at module level
            for func_name, func in program.functions.items():
                context.var_map[func_name] = {}
                for stmt in func.body:
                    if isinstance(stmt, VarDecl):
                        if isinstance(stmt.expr, StringExpr):
                            key = (func_name, stmt.expr.value)
                            if key not in context.string_map:
                                label = new_label()
                                context.string_map[key] = label
                                escaped_str = stmt.expr.value.replace("\n", "\\0A").replace('"', '\\22')
                                context.string_decls.append(f'{label} = private constant [{len(stmt.expr.value) + 1} x i8] c"{escaped_str}\\00"')
                    elif isinstance(stmt, Stdout):
                        if isinstance(stmt.expr, StringExpr):
                            key = (func_name, stmt.expr.value)
                            if key not in context.string_map:
                                label = new_label()
                                context.string_map[key] = label
                                escaped_str = stmt.expr.value.replace("\n", "\\0A").replace('"', '\\22')
                                context.string_decls.append(f'{label} = private constant [{len(stmt.expr.value) + 1} x i8] c"{escaped_str}\\00"')

            # Write string declarations
            for decl in context.string_decls:
                f.write(f"{decl}\n")
            f.write("\n")

            # Generate functions
            for func_name, func in program.functions.items():
                f.write(f'define void @{func_name}() {{\n')
                f.write('entry:\n')
                # Allocate variables and generate instructions
                for stmt in func.body:
                    if isinstance(stmt, VarDecl):
                        reg = new_reg()
                        llvm_type = map_rock_to_llvm_type(stmt.type)
                        context.var_map[func_name][stmt.name] = (llvm_type, reg)
                        f.write(f'  {reg} = alloca {llvm_type}\n')
                        expr_reg, expr_type = process_expr(stmt.expr, func_name, f)
                        f.write(f'  store {expr_type} {expr_reg}, {llvm_type}* {reg}\n')
                    elif isinstance(stmt, Stdout):
                        expr_reg, expr_type = process_expr(stmt.expr, func_name, f)
                        if expr_type == "i64":
                            f.write(f'  call i32 (i8*, ...) @printf(i8* getelementptr ([5 x i8], [5 x i8]* @.str.int, i32 0, i32 0), i64 {expr_reg})\n')
                        elif expr_type == "double":
                            f.write(f'  call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [5 x i8]* @.str.double, i32 0, i32 0), double {expr_reg})\n')
                        elif expr_type == "i8*":
                            f.write(f'  call i32 (i8*, ...) @printf(i8* {expr_reg})\n')
                        else:
                            log_error(f"Unsupported type '{expr_type}' for stdout in function '{func_name}'", None, ERROR_SUGGESTIONS["type_mismatch"])
                            sys.exit(1)
                    elif isinstance(stmt, Call):
                        if stmt.func_name not in program.functions:
                            log_error(f"Calling undefined function '{stmt.func_name}' in function '{func_name}'", None, ERROR_SUGGESTIONS["invoking_undefined_function"])
                            sys.exit(1)
                        f.write(f'  call void @{stmt.func_name}()\n')
                f.write('  ret void\n}\n\n')

            # Main function
            f.write('define i32 @main() {\n')
            f.write('entry:\n')
            if program.invoke:
                f.write(f'  call void @{program.invoke}()\n')
            f.write('  ret i32 0\n}\n')

        log_info("LLVM IR generation completed")
    except Exception as e:
        log_error(f"Failed to write LLVM IR file: {str(e)}")
        sys.exit(1)