import ipaddress
from urllib.parse import urlparse
import socket

def is_safe_url(url):
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return False

        # Block localhost by name
        if hostname in ["localhost", "127.0.0.1"]:
            return False

        # Resolve hostname to IP
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)

        # Block internal/private/reserved IPs
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
            return False

        return True
    except Exception:
        return False
