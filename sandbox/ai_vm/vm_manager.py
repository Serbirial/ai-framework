import os
import time
import requests
import uuid
import json
import threading
from firecracker import MicroVM

SESSIONS_DIR = "./user_state"
os.makedirs(SESSIONS_DIR, exist_ok=True)
METADATA_FILE = os.path.join(SESSIONS_DIR, "sessions.json")

KERNEL_PATH = os.path.join(SESSIONS_DIR, "vmlinux.bin")
BASE_ROOTFS_PATH = os.path.join(SESSIONS_DIR, "base_rootfs.ext4")

LOCK = threading.Lock()

try:
	with open(METADATA_FILE, "r") as f:
		sessions = json.load(f)
except Exception:
	sessions = {}

def persist_sessions():
	with LOCK:
		with open(METADATA_FILE, "w") as f:
			json.dump(sessions, f)

def safe_session_id(session_id):
	return ''.join(c for c in session_id if c.isalnum() or c in ('-', '_'))

def workspace_path(session_id):
	path = os.path.join(SESSIONS_DIR, safe_session_id(session_id))
	os.makedirs(path, exist_ok=True)
	return path

class FirecrackerManager:
	def __init__(self, max_active_vms=3, max_total_vms=5, inactivity_timeout=3600):
		self.vms = {}  # session_id -> MicroVM instance
		self.max_active_vms = max_active_vms
		self.max_total_vms = max_total_vms
		self.inactivity_timeout = inactivity_timeout

	def create_vm(self, session_id=None, vcpu=1, mem_mib=512):
		session_id = session_id or uuid.uuid4().hex
		ws = workspace_path(session_id)

		vm = MicroVM(
			name=session_id,
			kernel_file=KERNEL_PATH,
			base_rootfs=BASE_ROOTFS_PATH,
			vcpu=vcpu,
			mem_size_mib=mem_mib,
			working_dir=ws,
			verbose=False,
			expose_ports=False,
			bridge=False,
		)
		status = vm.create()
		if not status.get("success", False):
			raise RuntimeError(f"Failed to create VM: {status}")

		with LOCK:
			sessions[session_id] = {
				"last_access": time.time(), # last AI access
				"inactive": False, # not actively being used by AI
				"snapshot_tag": None,
				"vcpu": vcpu,
				"mem_mib": mem_mib,
				"loaded": True,  # means VM is fully loaded
			}
			persist_sessions()

		self.vms[session_id] = vm

		# Enforce limits after creation
		self._enforce_limits()

		return session_id

	def get_vm(self, session_id):
		if session_id in self.vms:
			return self.vms[session_id]

		if session_id not in sessions:
			return None

		ws = workspace_path(session_id)
		meta = sessions[session_id]

		vm = MicroVM(
			name=session_id,
			kernel_file=KERNEL_PATH,
			base_rootfs=BASE_ROOTFS_PATH,
			vcpu=meta.get("vcpu", 1),
			mem_size_mib=meta.get("mem_mib", 512),
			working_dir=ws,
			verbose=False,
			expose_ports=False,
			bridge=False,
		)

		snapshot_tag = meta.get("snapshot_tag")
		if snapshot_tag:
			vmstate_path = os.path.join(ws, f"{snapshot_tag}.vmstate")
			mem_path = os.path.join(ws, f"{snapshot_tag}.mem")

			# Firecracker API socket path, usually in vm working directory, e.g.
			api_socket = os.path.join(ws, "api.socket")


			import requests_unixsocket
			session = requests_unixsocket.Session()
			url = f"http+unix://{api_socket.replace('/', '%2F')}/snapshot/load"

			payload = {
				"snapshot_path": vmstate_path,
				"mem_file_path": mem_path,
				"resume_vm": True
			}

			try:
				response = session.put(url, json=payload)
				response.raise_for_status()
			except Exception as e:
				print(f"Failed to load snapshot for VM {session_id}: {e}")

			# After loading snapshot, start the VM (resume)
			start_url = f"http+unix://{api_socket.replace('/', '%2F')}/actions"
			start_payload = {"action_type": "InstanceStart"}

			try:
				response = session.put(start_url, json=start_payload)
				response.raise_for_status()
			except Exception as e:
				print(f"Failed to start VM {session_id} after snapshot load: {e}")

		with LOCK:
			sessions[session_id]["loaded"] = True
			persist_sessions()

		self.vms[session_id] = vm
		return vm


	def run_command(self, session_id, command, username="root", key_path=None):
		vm = self.get_vm(session_id)
		if not vm:
			raise ValueError("Session VM not found")

		with LOCK:
			sessions[session_id]["last_access"] = time.time()
			sessions[session_id]["inactive"] = False
			persist_sessions()

		status = vm.status()
		if status != "running":
			vm.resume()

		self._enforce_limits()

		ssh = vm.connect(username=username, key_path=key_path)
		stdin, stdout, stderr = ssh.exec_command(command)
		out = stdout.read().decode()
		err = stderr.read().decode()
		ssh.close()
		return out, err

	def delete_vm(self, session_id):
		vm = self.get_vm(session_id)
		if vm:
			vm.delete()
		with LOCK:
			sessions.pop(session_id, None)
			persist_sessions()
		ws = workspace_path(session_id)
		if os.path.exists(ws):
			import shutil
			shutil.rmtree(ws, ignore_errors=True)
		self.vms.pop(session_id, None)

	def snapshot_and_unload_vm(self, session_id):
		"""Pause, snapshot, delete VM to free resources, and unload from manager."""
		vm = self.get_vm(session_id)
		if not vm:
			return

		vm.pause()
		tag = f"snapshot_{int(time.time())}"
		vm.create_snapshot(tag)

		vm.delete() # fully shut down the VM

		with LOCK:
			sessions[session_id]["snapshot_tag"] = tag
			sessions[session_id]["loaded"] = False
			persist_sessions()

		# remove VM instance from loaded VM list
		self.vms.pop(session_id, None)


	def mark_inactive(self, session_id):
		with LOCK:
			if session_id in sessions:
				sessions[session_id]["inactive"] = True
				persist_sessions()
				self._enforce_limits()

	def get_status(self, session_id):
		vm = self.get_vm(session_id)
		if not vm:
			return None
		return vm.status()

	def list_sessions(self):
		with LOCK:
			return list(sessions.keys())


def _enforce_limits(self):
    with LOCK:
        running_vms = []
        for sid in sessions:
            vm = self.get_vm(sid)
            if vm and vm.status() == "running":
                inactive_flag = sessions[sid].get("inactive", False)
                running_vms.append((sid, inactive_flag, sessions[sid]["last_access"]))

        #  active (inactive=False) first, then newest last_access first
        running_vms.sort(key=lambda x: (x[1], -x[2]))

        active_count = sum(not x[1] for x in running_vms)

        for sid, inactive_flag, _ in running_vms:
            if active_count <= self.max_active_vms:
                break
            if inactive_flag:
                vm = self.get_vm(sid)
                if vm:
                    vm.pause()
                    active_count -= 1

        loaded_vms = [sid for sid in sessions if sessions[sid].get("loaded", True)]
        if len(loaded_vms) > self.max_total_vms:
            loaded_vms.sort(key=lambda sid: sessions[sid]["last_access"])
            excess = len(loaded_vms) - self.max_total_vms

            candidates = []
            for sid in loaded_vms:
                if excess <= 0:
                    break
                inactive_flag = sessions[sid].get("inactive", False)
                vm = self.vms.get(sid)
                if inactive_flag and vm:
                    status = vm.status()
                    if status != "running":
                        candidates.append(sid)
                        excess -= 1

            for sid in candidates:
                self.snapshot_and_unload_vm(sid)

