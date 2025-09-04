from rich import print
from Logger import log_info, log_error
from Compiler import rock_to_llvm, compile_llvm_to_exe
import sys
import os

# ----------------- TERMINAL -----------------
def Terminal():
    log_info("Starting Rock compiler terminal")
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