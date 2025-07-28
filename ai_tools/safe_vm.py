import requests

BASE_URL = "http://127.0.0.1:5000"  # Change if your Flask app runs elsewhere

headers = {
    "User-Agent": "AI/1.0 (RespectfulAI; +http://127.0.0.1/)",
    "Referer": "http://127.0.0.1/"
}

def create_session():
    try:
        res = requests.post(f"{BASE_URL}/session", headers=headers, timeout=12)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": f"Exception creating session: {str(e)}"}

def load_session(session_id):
    try:
        res = requests.post(f"{BASE_URL}/session/{session_id}/load", headers=headers, timeout=20)
        if res.status_code != 200:
            return {"error": f"Failed to load session: HTTP {res.status_code}", "details": res.json()}
        return res.json()
    except Exception as e:
        return {"error": f"Exception loading session: {str(e)}"}

def run_command_stream(session_id, command):
    try:
        with requests.post(
            f"{BASE_URL}/session/{session_id}/run-stream",
            json={"command": command},
            headers=headers,
            stream=True,
            timeout=None
        ) as res:
            if res.status_code != 200:
                return {"error": f"Failed to run command: HTTP {res.status_code}", "details": res.json()}
            for chunk in res.iter_content(chunk_size=None):
                print(chunk.decode(), end='')
            return {"status": "completed"}
    except Exception as e:
        return {"error": f"Exception running command: {str(e)}"}


def run_command(session_id, command):
    try:
        res = requests.post(f"{BASE_URL}/session/{session_id}/run", json={"command": command}, headers=headers, timeout=600)
        if res.status_code != 200:
            return {"error": f"Failed to run command: HTTP {res.status_code}", "details": res.json()}
        return res.json()
    except Exception as e:
        return {"error": f"Exception running command: {str(e)}"}

def get_status(session_id):
    try:
        res = requests.get(f"{BASE_URL}/session/{session_id}/status", headers=headers, timeout=12)
        if res.status_code != 200:
            return {"error": f"Failed to get status: HTTP {res.status_code}", "details": res.json()}
        return res.json()
    except Exception as e:
        return {"error": f"Exception getting status: {str(e)}"}

EXPORT = {
    "create_session": {
        "help": "Create a new VM session and return its session ID.",
        "callable": create_session,
        "params": {}
    },
    "load_session": {
        "help": "Load a saved VM session by session ID.",
        "callable": load_session,
        "params": {
            "session_id": "The ID of the session to load"
        }
    },
    "run_command": {
        "help": "Run a shell command inside the VM session.",
        "callable": run_command,
        "params": {
            "session_id": "The ID of the session",
            "command": "The shell command string to run"
        }
    },
    "get_status": {
        "help": "Get the current status of the VM session.",
        "callable": get_status,
        "params": {
            "session_id": "The ID of the session"
        }
    },
}

if __name__ == "__main__":
    import json

    # Example usage:
    print("=== Create Session ===")
    new_session = create_session()
    print(json.dumps(new_session, indent=2))

    if "session_id" in new_session:
        sid = new_session["session_id"]

        print(f"\n=== Load Session {sid} ===")
        loaded = load_session(sid)
        print(json.dumps(loaded, indent=2))

        print(f"\n=== Run Command on Session {sid} ===")
        output = run_command(sid, "echo Hello from VM")
        print(json.dumps(output, indent=2))

        print(f"\n=== Get Status of Session {sid} ===")
        status = get_status(sid)
        print(json.dumps(status, indent=2))

        print(f"\n=== Delete Session {sid} ===")
        deleted = delete_session(sid)
        print(json.dumps(deleted, indent=2))
