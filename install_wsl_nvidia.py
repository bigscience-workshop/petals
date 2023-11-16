import subprocess
import time

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def check_command_output(command):
    try:
        output = subprocess.check_output(command, shell=True, text=True)
        return output.strip()
    except subprocess.CalledProcessError:
        return None

def is_wsl2_installed():
    version_output = check_command_output("wsl --list --verbose")
    return version_output and "2" in version_output

def install_wsl():
    if not is_wsl2_installed():
        print("Installing WSL 2...")
        run_command("wsl --install")
    else:
        print("WSL 2 is already installed.")

def has_nvidia_gpu():
    nvidia_output = check_command_output("wsl nvidia-smi")
    return "NVIDIA" in nvidia_output

def install_python():
    python_output = check_command_output("wsl python --version")
    if not python_output:
        print("Installing basic Python packages in WSL...")
        run_command("wsl sudo apt update")
        run_command("wsl sudo apt install python3-pip python-is-python3")
    else:
        print("Python is already installed in WSL.")

# def check_command_output(command):
#     try:
#         process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         output, _ = process.communicate()
#         return output.strip()
#     except subprocess.CalledProcessError:
#         return None

def has_petals_installed():
    petals_output = check_command_output("wsl python -m petals.cli.run_server -v")
    # Pause for 3 seconds
    time.sleep(3)
    return petals_output is not None

# def has_petals_installed2():
#     command = ["wsl", "python", "-m", "petals.cli.run_server"]
#     petals_output = check_command_output(command)
#     has_petals_installed()
#     return petals_output is not None

def install_petals():
    if not has_petals_installed():
        print("Installing Petals in WSL...")
        run_command("wsl python -m pip install git+https://github.com/bigscience-workshop/petals")
    else:
        print("Petals is already installed in WSL.")

def run_petals_server():
    print("Running Petals server...")
    run_command("wsl python -m petals.cli.run_server petals-team/StableBeluga2")

if __name__ == "__main__":
    install_wsl()
    
    if not has_nvidia_gpu():
        print("No NVIDIA GPU available in WSL. Exiting.")
        exit(1)
    
    install_python()
    install_petals()
    # Add a command to keep the WSL shell open
    subprocess.run(["wsl", "tail", "-f", "/dev/null"])
    #run_petals_server()