import platform
import socket
import psutil
import json
import sys

def get_cpu_model() -> str:
    # Try platform.processor() first
    cpu = platform.processor()
    if cpu and cpu != "x86_64" and cpu != "AMD64" and cpu != "i386":
        return cpu.strip()

    # On Linux, try /proc/cpuinfo for a more descriptive model name
    if sys.platform == "linux" or sys.platform == "linux2":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    # On macOS try sysctl
    if sys.platform == "darwin":
        try:
            import subprocess
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

    # Fallback to platform.machine()
    return platform.machine()

def get_system_info() -> dict:
    try:
        info = {}
        info['os'] = platform.system()
        info['os_version'] = platform.version()
        info['platform'] = platform.platform()
        info['hostname'] = socket.gethostname()
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['cpu_model'] = get_cpu_model()
        mem = psutil.virtual_memory()
        info['total_ram_gb'] = round(mem.total / (1024**3), 2)
        info['python_version'] = platform.python_version()
        return {"result": info}
    except Exception as e:
        return {"error": f"Failed to get system info: {str(e)}"}

EXPORT = {
    "get_system_info": {
        "help": "Get the current machine's basic system information (OS, CPU model and cores, RAM, hostname, Python version).",
        "callable": get_system_info,
        "params": {}
    }
}

if __name__ == "__main__":
    import json
    print(json.dumps(get_system_info(), indent=2))
