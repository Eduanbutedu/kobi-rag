"""Repair incomplete TLS certificate chains without disabling verification.

Several Turkish government sites (mevzuat.gov.tr among them) send only their
leaf certificate and omit the intermediate CA. Browsers recover by fetching
the issuer named in the certificate's Authority Information Access
extension; Python's ssl module does not, so verification fails with
"unable to get local issuer certificate".

This module does what the browser does: fetch the named intermediate and add
it to a copy of the certifi bundle. Verification stays enabled and the trust
anchor is still the root already in certifi -- the only thing supplied is the
intermediate the server should have sent.
"""

import re
import socket
import ssl
import tempfile
from pathlib import Path

import certifi
import requests

# AIA caIssuers alanındaki sertifika URL'si; DER içinde düz ASCII olarak duruyor
_CA_ISSUER_URL = re.compile(rb"http://[A-Za-z0-9./_\-]+\.(?:crt|cer|p7c)")
HANDSHAKE_TIMEOUT = 30
FETCH_TIMEOUT = 30
PEM_HEADER = b"-----BEGIN"


def extract_ca_issuer_url(der: bytes) -> str | None:
    """Find the issuing CA's certificate URL inside a DER-encoded certificate."""
    match = _CA_ISSUER_URL.search(der)
    return match.group().decode("ascii") if match else None


def to_pem(data: bytes) -> str:
    """Normalise a certificate to PEM, accepting either DER or PEM input."""
    if data.lstrip().startswith(PEM_HEADER):
        return data.decode("ascii")
    return ssl.DER_cert_to_PEM_cert(data)


def build_bundle(intermediate_pem: str, destination: Path) -> Path:
    """Write a CA bundle of certifi's roots plus one extra intermediate."""
    roots = Path(certifi.where()).read_text(encoding="utf-8")
    destination.write_text(f"{roots}\n{intermediate_pem}", encoding="utf-8")
    return destination


def peer_certificate(host: str, port: int = 443) -> bytes:
    """Fetch the server's leaf certificate without validating it.

    Validation is exactly what is broken here, so it is skipped for this one
    read. Nothing is trusted on the basis of these bytes: they are only used
    to look up which intermediate to fetch, and that intermediate must still
    chain to a root in certifi for the real request to succeed.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=HANDSHAKE_TIMEOUT) as sock:
        with context.wrap_socket(sock, server_hostname=host) as tls:
            return tls.getpeercert(binary_form=True)


class ChainRepair:
    """Per-host cache of repaired CA bundles."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix="kobi-ca-"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._bundles: dict[str, str] = {}

    def verify_for(self, host: str) -> str | bool:
        """The `verify` argument to pass to requests for this host."""
        return self._bundles.get(host, True)

    def repair(self, host: str) -> bool:
        """Try to build a complete bundle for this host. True if it worked."""
        if host in self._bundles:
            return True
        try:
            der = peer_certificate(host)
            issuer_url = extract_ca_issuer_url(der)
            if not issuer_url:
                return False
            response = requests.get(issuer_url, timeout=FETCH_TIMEOUT)
            response.raise_for_status()
            bundle = build_bundle(to_pem(response.content), self._cache_dir / f"{host}.pem")
        except (OSError, ValueError, requests.RequestException):
            return False
        self._bundles[host] = str(bundle)
        return True
