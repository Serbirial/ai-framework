import requests

SANDBOX_API_URL = "http://192.168.0.9:5000/run_code"  # Adjust port/path as needed

def run_code_sandboxed(user_code: str) -> dict:
    """
    Sends user_code to a remote sandbox API (on 192.168.0.8) which runs it safely inside Docker,
    then returns the execution result or error.
    """
    try:
        response = requests.post(
            SANDBOX_API_URL,
            json={"user_code": user_code},
            timeout=30
        )
        if response.status_code != 200:
            return {
                "status": "error",
                "message": f"Sandbox API returned status code {response.status_code}"
            }
        data = response.json()
        # Expecting data dict with keys like 'status', 'output', 'message'
        return data

    except requests.Timeout:
        return {"status": "error", "message": "Sandbox API request timed out (30s limit)"}
    except requests.ConnectionError:
        return {"status": "error", "message": "Failed to connect to sandbox API (might be down or in use)"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error contacting sandbox API: {str(e)}"}

EXPORT = {
    "run_code_sandboxed": {
        "help": "Run Python code sandboxed remotely via API  (500M RAM, NO NETWORK 25S TIMEOUT)",
        "callable": run_code_sandboxed,
        "params": {
            "user_code": "string, required — Python code to execute safely."
        }
    }
}

if __name__ == "__main__":
    import json
    test_code = "print('Hello from remote sandbox!')"
    print(json.dumps(run_code_sandboxed(test_code), indent=2))
