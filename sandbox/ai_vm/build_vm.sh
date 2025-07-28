#!/bin/bash
# Usage:
#   sudo ./build_vm.sh /path/to/public_key.pub /output/dir

set -e

PUB_KEY_PATH=$1
OUT_DIR=$2

if [ -z "$PUB_KEY_PATH" ] || [ -z "$OUT_DIR" ]; then
  echo "Usage: sudo $0 /path/to/id_rsa.pub /output/dir"
  exit 1
fi

if [ ! -f "$PUB_KEY_PATH" ]; then
  echo "Public key file not found: $PUB_KEY_PATH"
  exit 1
fi

# Create working directories
mkdir -p "$OUT_DIR/rootfs"
ROOTFS_DIR="$OUT_DIR/rootfs"

echo "Bootstrapping minimal Debian rootfs into $ROOTFS_DIR ..."

debootstrap --variant=minbase \
  --include=python3,python3-pip,python3-venv,build-essential,\
git,curl,wget,ca-certificates,net-tools,iproute2,iputils-ping,\
busybox,openssh-server,gcc,g++,make,libssl-dev,libffi-dev,pkg-config,\
go,nodejs,npm,sqlite3,libsqlite3-dev \
  stable "$ROOTFS_DIR" http://deb.debian.org/debian


echo "Configuring SSH server and authorized keys..."

# Create root ssh folder
mkdir -p "$ROOTFS_DIR/root/.ssh"
chmod 700 "$ROOTFS_DIR/root/.ssh"
cp "$PUB_KEY_PATH" "$ROOTFS_DIR/root/.ssh/authorized_keys"
chmod 600 "$ROOTFS_DIR/root/.ssh/authorized_keys"

# Enable root login via SSH (permit root login with keys only)
echo "PermitRootLogin prohibit-password" > "$ROOTFS_DIR/etc/ssh/sshd_config.d/01-root.conf"

# Setup systemd to start sshd by default
echo '/usr/sbin/sshd' > "$ROOTFS_DIR/etc/rc.local"
chmod +x "$ROOTFS_DIR/etc/rc.local"


# Fix resolv.conf for DNS
echo "nameserver 8.8.8.8" > "$ROOTFS_DIR/etc/resolv.conf"

# Setup minimal fstab
cat <<EOF > "$ROOTFS_DIR/etc/fstab"
proc /proc proc defaults 0 0
sysfs /sys sysfs defaults 0 0
devpts /dev/pts devpts defaults 0 0
tmpfs /tmp tmpfs defaults 0 0
EOF

# Setup minimal hostname
echo "firecracker-vm" > "$ROOTFS_DIR/etc/hostname"

# Prepare /etc/hosts
cat <<EOF > "$ROOTFS_DIR/etc/hosts"
127.0.0.1 localhost
127.0.1.1 firecracker-vm

# The following lines are desirable for IPv6 capable hosts
::1 ip6-localhost ip6-loopback
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
EOF

echo "Creating ext4 image file from rootfs..."

IMG_SIZE=4G
IMG_PATH="$OUT_DIR/rootfs.ext4"

# Create empty image
dd if=/dev/zero of="$IMG_PATH" bs=1M count=0 seek=2000

# Format as ext4
mkfs.ext4 -F "$IMG_PATH"

# Mount image and copy files
mkdir -p /tmp/firecracker_img_mount
sudo mount -o loop "$IMG_PATH" /tmp/firecracker_img_mount
sudo cp -a "$ROOTFS_DIR/." /tmp/firecracker_img_mount/
sync
sudo umount /tmp/firecracker_img_mount
rmdir /tmp/firecracker_img_mount

echo "Done. Rootfs image created at $IMG_PATH"
echo "Make sure you have your kernel image ready (vmlinux.bin) in $OUT_DIR"

