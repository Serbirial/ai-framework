from flask import Flask, request, jsonify
import os, uuid, json, time, shutil
import docker

app = Flask(__name__)
client = docker.from_env()

# Configuration
SESSION_ROOT = "/var/sandbox_vms"
METADATA_FILE = os.path.join(SESSION_ROOT, "sessions.json")
# Default image name for sandbox VMs; build locally with:
#   docker build -t safe-vm:latest ./full_vm
DEFAULT_IMAGE = "safe-vm:latest"
RESOURCE_LIMITS = {"mem_limit": "512m", "cpu_quota": 50000}

os.makedirs(SESSION_ROOT, exist_ok=True)

# Load or initialize session metadata
# sessions: session_id -> { "snapshot_tag": str, "last_access": float }
try:
    with open(METADATA_FILE) as f:
        sessions = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sessions = {}


def persist_sessions():
    """Save session metadata to disk."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(sessions, f)


def get_workspace_path(session_id):
    """Return (and create) workspace directory for session."""
    safe = ''.join(c for c in session_id if c.isalnum() or c in ('-','_'))
    path = os.path.join(SESSION_ROOT, safe)
    os.makedirs(path, exist_ok=True)
    return path


def get_container_name(session_id):
    return f"sandbox_{session_id}"


def ensure_container(session_id):
    """Return the existing container for session or create a new one from snapshot_tag."""
    meta = sessions.get(session_id)
    if not meta:
        return None
    name = get_container_name(session_id)
    try:
        return client.containers.get(name)
    except docker.errors.NotFound:
        # Create fresh container from snapshot_tag
        workspace = get_workspace_path(session_id)
        return client.containers.create(
            image=meta['snapshot_tag'],
            command="sleep infinity",
            name=name,
            tty=True,
            network_mode='host',  # full internet
            volumes={workspace: {'bind': '/workspace', 'mode': 'rw'}},
            working_dir='/workspace',
            **RESOURCE_LIMITS
        )

@app.route('/session', methods=['POST'])
def create_session():
    """
    Create a new sandbox session. Accepts optional JSON:
      - base_image: Docker image to use instead of DEFAULT_IMAGE
    Returns:
      session_id (str), base_image (str)
    """
    data = request.get_json(silent=True) or {}
    session_id = uuid.uuid4().hex
    # Determine initial image
    base_image = data.get('base_image', DEFAULT_IMAGE)
    # Setup metadata
    sessions[session_id] = {
        'snapshot_tag': base_image,
        'last_access': time.time()
    }
    persist_sessions()
    # Prepare workspace
    get_workspace_path(session_id)
    return jsonify({'session_id': session_id, 'base_image': base_image}), 201

@app.route('/session/<session_id>/run/python', methods=['POST'])
def run_python(session_id):
    """
    Execute Python code in the session's VM.
    Body JSON: { 'code': '<python code>' }
    """
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session_id'}), 404
    data = request.get_json(force=True) or {}
    code = data.get('code')
    if not code:
        return jsonify({'error': 'Missing code'}), 400

    # Update last access timestamp
    sessions[session_id]['last_access'] = time.time()
    persist_sessions()

    # Ensure container exists
    container = ensure_container(session_id)
    if not container:
        return jsonify({'error': 'Failed to create container'}), 500

    # Write code to workspace
    ws = get_workspace_path(session_id)
    script_path = os.path.join(ws, 'temp_script.py')
    with open(script_path, 'w') as f:
        f.write(code)

    try:
        container.reload()
        if container.status != 'running':
            container.start()
        res = container.exec_run(
            cmd=["python3", "script.py"],
            workdir='/workspace', stdout=True, stderr=True, demux=True, timeout=120
        )
        stdout, stderr = res.output if hasattr(res, 'output') else res
        output = (stdout or b'').decode() + (stderr or b'').decode()
    except Exception as e:
        output = f"Error: {e}"
    finally:
        # Stop to free resources
        try:
            container.stop(timeout=5)
        except:
            pass

    return jsonify({'output': output}), 200

@app.route('/session/<session_id>/snapshot', methods=['POST'])
def snapshot_session(session_id):
    """
    Commit container state to a new image tag.
    Body JSON: { 'tag': '<image_name>' }
    Returns snapshot_image id.
    """
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session_id'}), 404
    data = request.get_json(force=True) or {}
    tag = data.get('tag')
    if not tag:
        return jsonify({'error': 'Missing tag'}), 400

    container = ensure_container(session_id)
    if not container:
        return jsonify({'error': 'Container not found'}), 404

    container.reload()
    if container.status == 'running':
        container.stop(timeout=5)

    img = container.commit(repository=tag)
    sessions[session_id]['snapshot_tag'] = tag
    sessions[session_id]['last_access'] = time.time()
    persist_sessions()

    return jsonify({'snapshot_image': img.id}), 200

@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """
    Delete the session, remove container and workspace.
    """
    meta = sessions.pop(session_id, None)
    if not meta:
        return jsonify({'error': 'Invalid session_id'}), 404
    persist_sessions()

    # Remove container
    name = get_container_name(session_id)
    try:
        client.containers.get(name).remove(force=True)
    except:
        pass
    # Cleanup workspace
    shutil.rmtree(get_workspace_path(session_id), ignore_errors=True)
    return jsonify({'deleted': session_id}), 200

@app.route('/session/<session_id>/status', methods=['GET'])
def session_status(session_id):
    """
    Return session metadata and container status.
    """
    meta = sessions.get(session_id)
    if not meta:
        return jsonify({'error': 'Invalid session_id'}), 404
    try:
        c = client.containers.get(get_container_name(session_id))
        state = c.status
    except:
        state = 'not_created'

    return jsonify({
        'session_id': session_id,
        'base_image': meta['snapshot_tag'],
        'last_access': meta['last_access'],
        'container_status': state
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
