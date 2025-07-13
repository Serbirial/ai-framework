import requests
import re

SANDBOX_API_URL = "http://192.168.0.9:5000/run_code"  # Adjust port/path as needed

def extract_code_from_md(user_code: str) -> str:
    """
    Extract Python code from a markdown code block if present.
    Example:
    ```python
    print("hello")
    ```
    returns just:
    print("hello")
    """
    # Regex to match ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(pattern, user_code, re.DOTALL | re.IGNORECASE)
    if matches:
        # Return the first matched code block content trimmed
        return matches[0].strip()
    return user_code.strip()

def run_code_sandboxed(user_code) -> dict:
    """
    Sends user_code to a remote sandbox API which runs it safely inside Docker,
    then returns the execution result or error.
    Supports extraction of python code from markdown code blocks.
    """
    if not isinstance(user_code, str):
        return {
            "status": "error",
            "message": "The 'code' parameter must be a string containing Python code."
        }

    # Extract python code from markdown if needed
    cleaned_code = extract_code_from_md(user_code)

    try:
        response = requests.post(
            SANDBOX_API_URL,
            json={"user_code": cleaned_code},
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
    "run_python_sandboxed": {
        "help": "Run Python code sandboxed remotely via API  (500M RAM, NO NETWORK 25S TIMEOUT)",
        "callable": run_code_sandboxed,
        "params": {
            "code": "string, required — Python code to execute safely."
        }
    }
}

if __name__ == "__main__":
    import json
    test_code = """
```python
print('Hello from remote sandbox!')
"""
    print(json.dumps(run_code_sandboxed(test_code), indent=2))