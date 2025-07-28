from .interactive_tool_class import InteractiveTool
import requests
from datetime import datetime
from src.static import VM_HOST_IP_PORT, WorkerConfig

class VmManagerInteractiveTool(InteractiveTool):
    """
    Interactive VM Manager tool with fixed session created once on init.
    Commands:
    - STATUS
    - RUN <shell_command>
    - FILE_READ <path>
    - FILE_WRITE <path> <content>
    - FILE_LIST <directory>
    - PKG_LIST
    - PKG_INSTALL <pkg1> [pkg2 ...]
    - PKG_UNINSTALL <pkg1> [pkg2 ...]
    - ENV_GET
    - ENV_SET <key> <value>
    - PROCESS_LIST
    - PROCESS_KILL <pid>
    - GIT <command> [args...]
    - LOGS <log_file> <lines>
    - DISK
    - EXIT
    - SCRIPT <commands>
    """

    def __init__(self, worker_config: WorkerConfig, base_url=VM_HOST_IP_PORT, **kwargs):
        super().__init__(**kwargs)
        self.worker_config = worker_config
        self.base_url = base_url.rstrip("/")
        self.identifier = worker_config.identifier
        self._input_queue = []
        self.detailed_logs = []
        self.done = False

        # Create session once on init
        resp = requests.post(f"{self.base_url}/session", json={"identifier": self.identifier})
        if resp.ok:
            data = resp.json()
            self.session_id = data.get("session_id")
            self.existing = data.get("existing", False)
        else:
            raise RuntimeError(f"Failed to create session: {resp.status_code} {resp.text}")

    def log_step(self, command, result):
        self.detailed_logs.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step": len(self.detailed_logs) + 1,
            "command": command,
            "result": result
        })

    def send_input(self, input_data):
        self._input_queue.append(input_data)

    def _api_post(self, path, json=None):
        url = f"{self.base_url}{path}"
        return requests.post(url, json=json)

    def _api_get(self, path, params=None):
        url = f"{self.base_url}{path}"
        return requests.get(url, params=params)

    def receive_output(self):
        if not self._input_queue:
            return "No command queued."

        cmd = self._input_queue.pop(0).strip()
        if not cmd:
            return "Empty command."

        parts = cmd.split(maxsplit=2)
        action = parts[0].upper()
        arg1 = parts[1] if len(parts) > 1 else None
        arg2 = parts[2] if len(parts) > 2 else None

        if action == "SCRIPT":
            lines = cmd[len("SCRIPT"):].strip().splitlines()
            for line in reversed(lines):
                if line.strip():
                    self._input_queue.insert(0, line.strip())
            result = f"Batched {len(lines)} commands."
            self.log_step(cmd, result)
            return result

        try:
            if action == "STATUS":
                resp = self._api_get(f"/session/{self.session_id}/status")
                if resp.ok:
                    data = resp.json()
                    result = f"Session status: {data.get('status')}"
                else:
                    result = f"Error: {resp.text}"

            elif action == "RUN":
                if not arg1:
                    result = "Usage: RUN <shell_command>"
                else:
                    resp = self._api_post(f"/session/{self.session_id}/run", json={"command": arg1})
                    if resp.ok:
                        data = resp.json()
                        out = data.get("stdout", "")
                        err = data.get("stderr", "")
                        result = f"STDOUT:\n{out}\nSTDERR:\n{err}"
                    else:
                        result = f"Error: {resp.text}"

            elif action == "FILE_READ":
                if not arg1:
                    result = "Usage: FILE_READ <path>"
                else:
                    resp = self._api_get(f"/session/{self.session_id}/file", params={"path": arg1})
                    if resp.ok:
                        data = resp.json()
                        result = f"File content:\n{data.get('content', '')}"
                    else:
                        result = f"Error: {resp.text}"

            elif action == "FILE_WRITE":
                if not arg1 or not arg2:
                    result = "Usage: FILE_WRITE <path> <content>"
                else:
                    resp = self._api_post(f"/session/{self.session_id}/file", json={"path": arg1, "content": arg2})
                    if resp.ok:
                        result = f"Wrote to {arg1}"
                    else:
                        result = f"Error: {resp.text}"

            elif action == "FILE_LIST":
                path = arg1 or "."
                resp = self._api_get(f"/session/{self.session_id}/files", params={"path": path})
                if resp.ok:
                    data = resp.json()
                    result = f"Files in {path}:\n{data.get('listing', '')}"
                else:
                    result = f"Error: {resp.text}"

            elif action == "PKG_LIST":
                resp = self._api_get(f"/session/{self.session_id}/packages")
                if resp.ok:
                    data = resp.json()
                    result = "Installed packages:\n" + "\n".join(data.get("packages", []))
                else:
                    result = f"Error: {resp.text}"

            elif action == "PKG_INSTALL":
                if not arg1:
                    result = "Usage: PKG_INSTALL <pkg1> [pkg2 ...]"
                else:
                    packages = arg1.split()
                    resp = self._api_post(f"/session/{self.session_id}/packages", json={"packages": packages})
                    if resp.ok:
                        data = resp.json()
                        result = f"Installed packages output:\n{data.get('stdout','')}\nErrors:\n{data.get('stderr','')}"
                    else:
                        result = f"Error: {resp.text}"

            elif action == "PKG_UNINSTALL":
                if not arg1:
                    result = "Usage: PKG_UNINSTALL <pkg1> [pkg2 ...]"
                else:
                    packages = arg1.split()
                    resp = requests.delete(f"{self.base_url}/session/{self.session_id}/packages", json={"packages": packages})
                    if resp.ok:
                        data = resp.json()
                        result = f"Uninstalled packages output:\n{data.get('stdout','')}\nErrors:\n{data.get('stderr','')}"
                    else:
                        result = f"Error: {resp.text}"

            elif action == "ENV_GET":
                resp = self._api_get(f"/session/{self.session_id}/env")
                if resp.ok:
                    data = resp.json()
                    env_vars = data.get("environment", {})
                    result = "\n".join(f"{k}={v}" for k, v in env_vars.items())
                else:
                    result = f"Error: {resp.text}"

            elif action == "ENV_SET":
                if not arg1 or not arg2:
                    result = "Usage: ENV_SET <key> <value>"
                else:
                    resp = self._api_post(f"/session/{self.session_id}/env", json={"key": arg1, "value": arg2})
                    if resp.ok:
                        result = f"Set environment variable {arg1}"
                    else:
                        result = f"Error: {resp.text}"

            elif action == "PROCESS_LIST":
                resp = self._api_get(f"/session/{self.session_id}/processes")
                if resp.ok:
                    data = resp.json()
                    result = "Processes:\n" + data.get("processes", "")
                else:
                    result = f"Error: {resp.text}"

            elif action == "PROCESS_KILL":
                if not arg1:
                    result = "Usage: PROCESS_KILL <pid>"
                else:
                    resp = requests.delete(f"{self.base_url}/session/{self.session_id}/processes", json={"pid": arg1})
                    if resp.ok:
                        data = resp.json()
                        result = f"Process kill output:\nSTDOUT:\n{data.get('stdout','')}\nSTDERR:\n{data.get('stderr','')}"
                    else:
                        result = f"Error: {resp.text}"

            elif action == "GIT":
                if not arg1:
                    result = "Usage: GIT <command> [args...]"
                else:
                    git_command = arg1
                    git_args = arg2.split() if arg2 else []
                    resp = self._api_post(f"/session/{self.session_id}/git", json={"command": git_command, "args": git_args})
                    if resp.ok:
                        data = resp.json()
                        result = f"GIT command output:\nSTDOUT:\n{data.get('stdout','')}\nSTDERR:\n{data.get('stderr','')}"
                    else:
                        result = f"Error: {resp.text}"

            elif action == "LOGS":
                log_file = arg1 or "/var/log/syslog"
                lines = arg2 or "100"
                try:
                    lines = int(lines)
                except ValueError:
                    lines = 100
                resp = self._api_get(f"/session/{self.session_id}/logs", params={"log_file": log_file, "lines": lines})
                if resp.ok:
                    data = resp.json()
                    result = f"Last {lines} lines of {log_file}:\n{data.get('content', '')}"
                else:
                    result = f"Error: {resp.text}"

            elif action == "DISK":
                resp = self._api_get(f"/session/{self.session_id}/disk")
                if resp.ok:
                    data = resp.json()
                    result = f"Available disk space: {data.get('available_mb')} MB"
                else:
                    result = f"Error: {resp.text}"

            elif action == "EXIT":
                self.done = True
                result = "Session terminated by EXIT."

            else:
                result = f"Unknown command '{action}'."

        except Exception as e:
            result = f"Error processing command '{cmd}': {e}"

        self.log_step(cmd, result)
        return result

    def describe(self) -> dict:
        return {
            "name": "VmManagerInteractiveTool",
            "description": (
                "Interactive VM Manager tool. Creates a dedicated session once on init.\n"
                "Use commands to run shell, file ops, package management, env vars, process control, git, logs, disk info.\n"
                "Batch commands with SCRIPT block.\n"
                "Session id: " + (self.session_id or "None")
            ),
            "commands": [
                "STATUS",
                "RUN <shell_command>",
                "FILE_READ <path>",
                "FILE_WRITE <path> <content>",
                "FILE_LIST <directory>",
                "PKG_LIST",
                "PKG_INSTALL <pkg1> [pkg2 ...]",
                "PKG_UNINSTALL <pkg1> [pkg2 ...]",
                "ENV_GET",
                "ENV_SET <key> <value>",
                "PROCESS_LIST",
                "PROCESS_KILL <pid>",
                "GIT <command> [args...]",
                "LOGS <log_file> <lines>",
                "DISK",
                "EXIT",
                "SCRIPT <commands>"
            ],
            "state_summary": {
                "session_id": self.session_id,
                "identifier": self.identifier,
                "queued_commands_count": len(self._input_queue),
                "logged_steps_count": len(self.detailed_logs),
            }
        }

    def cleanup(self):
        # Could optionally delete the session here if API supports it.
        pass


if __name__ == "__main__":
    print("VmManagerInteractiveTool REPL")
    base_url = input("Enter VM Manager API base URL (default http://localhost:5001): ").strip() or "http://localhost:5001"
    identifier = input("Enter session identifier (default interactive_tool_session): ").strip() or "interactive_tool_session"
    try:
        tool = VmManagerInteractiveTool(base_url=base_url, identifier=identifier)
        print(f"Session created with id: {tool.session_id}")
    except Exception as e:
        print(f"Failed to create session: {e}")
        exit(1)
    print(tool.describe())

    try:
        while not tool.done:
            user_input = input(">> ").strip()
            if user_input.upper() == "SCRIPT":
                print("Enter SCRIPT commands line by line. Empty line to end.")
                lines = []
                while True:
                    line = input()
                    if not line.strip():
                        break
                    lines.append(line)
                script_block = "SCRIPT\n" + "\n".join(lines)
                tool.send_input(script_block)
            else:
                tool.send_input(user_input)

            output = tool.receive_output()
            if isinstance(output, list):
                for i, item in enumerate(output):
                    print(f"[{i}] {item}")
            else:
                print(output)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, exiting.")

    finally:
        tool.cleanup()
        print("VmManagerInteractiveTool session ended.")
