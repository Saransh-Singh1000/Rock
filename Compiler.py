from rich import print
import subprocess
import os
import sys
import tempfile
from Logger import CLANG_PATH, GCC_PATH, log_info, log_error, log_warning
from Tokenizer import tokenize
from Parser import parse
from LLVMGenrator import generate_llvm_ir


# ----------------- COMPILE LLVM TO EXE -----------------
def compile_llvm_to_exe(ll_file: str, exe_name: str):
    log_info(f"Compiling LLVM IR to executable: {exe_name}")
    exe_file = os.path.join(os.getcwd(), exe_name)
    obj_file = os.path.splitext(exe_file)[0] + ".o"
    try:
        # Compile LLVM IR to object file
        result = subprocess.run([CLANG_PATH, "-c", ll_file, "-o", obj_file], capture_output=True, text=True, check=True)
        log_info(f"Object file created: {obj_file}")
        # Link object file to executable
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