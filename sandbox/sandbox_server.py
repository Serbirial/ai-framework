from flask import Flask, request, jsonify
import subprocess
import uuid
import os
import tempfile

app = Flask(__name__)

def run_code_sandboxed(user_code: str):
    # Create a unique temp file path for the user code
    temp_code_path = os.path.join(tempfile.gettempdir(), f"run_{uuid.uuid4().hex}.py")

    # Write the user code to the temp file
    with open(temp_code_path, "w") as f:
        f.write(user_code)

    try:
        # Run the docker container, mounting the temp file as /app/run.py (read-only)
        result = subprocess.run([
            "docker", "run", "--rm",
            "--memory=275", "--cpus=0.5",  # Enforce resource limits
            "-v", f"{temp_code_path}:/app/run.py:ro",  # Mount user code inside container
            "--network=none",  # Disable networking for safety
            "safe-py-sandbox"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=25  # Overall timeout for code execution
        )

        # Decode and return output
        return {
            "status": "ok",
            "output": result.stdout.decode("utf-8")
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Code execution timed out (25s limit)"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Run failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}
    finally:
        # Clean up temp file no matter what
        if os.path.exists(temp_code_path):
            os.remove(temp_code_path)

@app.route('/run_code', methods=['POST'])
def api_run_code():
    data = request.get_json(force=True)
    if not data or 'user_code' not in data:
        return jsonify({"status": "error", "message": "Missing 'user_code' in JSON body."}), 400

    user_code = data['user_code']
    result = run_code_sandboxed(user_code)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
