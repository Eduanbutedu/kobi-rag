import ssl
from pathlib import Path

import certifi
import pytest

from eval.tls_chain import ChainRepair, build_bundle, extract_ca_issuer_url, to_pem

# Gerçek bir sertifikanın DER'i içinde AIA URL'si düz ASCII olarak duruyor
DER_WITH_AIA = (
    b"\x30\x82\x01\x0a" + b"junk" * 10 + b"http://cacerts.geotrust.com/GeoTrustTLSRSACAG1.crt"
    b"\x00\x01binary tail"
)

PEM_TEXT = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBkTCB+wIJAKt2ZQ0Y0000MA0GCSqGSIb3DQEBCwUAMBQxEjAQBgNVBAMMCWxv\n"
    "-----END CERTIFICATE-----\n"
)


def test_extracts_the_ca_issuer_url():
    assert (
        extract_ca_issuer_url(DER_WITH_AIA)
        == "http://cacerts.geotrust.com/GeoTrustTLSRSACAG1.crt"
    )


@pytest.mark.parametrize("suffix", [b".crt", b".cer", b".p7c"])
def test_recognises_each_certificate_extension(suffix):
    der = b"\x30\x82" + b"http://ca.example.com/inter" + suffix + b"\x00"
    assert extract_ca_issuer_url(der).endswith(suffix.decode())


def test_returns_none_when_there_is_no_aia():
    assert extract_ca_issuer_url(b"\x30\x82 no urls here \x00\x01") is None


def test_ignores_urls_that_are_not_certificates():
    # OCSP URL'si (http://status.geotrust.com) ara sertifika değildir
    der = b"\x30\x82http://status.geotrust.com\x00"
    assert extract_ca_issuer_url(der) is None


def test_pem_input_is_passed_through():
    assert to_pem(PEM_TEXT.encode("ascii")) == PEM_TEXT


def test_der_input_is_converted_to_pem():
    # Kendi ürettiğimiz sertifikayı DER'e çevirip geri okuyoruz
    der = ssl.PEM_cert_to_DER_cert(PEM_TEXT)
    converted = to_pem(der)
    assert converted.startswith("-----BEGIN CERTIFICATE-----")
    assert ssl.PEM_cert_to_DER_cert(converted) == der


def test_bundle_contains_certifi_roots_plus_the_intermediate(tmp_path):
    destination = build_bundle(PEM_TEXT, tmp_path / "bundle.pem")
    content = destination.read_text(encoding="utf-8")

    assert PEM_TEXT in content
    assert Path(certifi.where()).read_text(encoding="utf-8") in content
    assert content.count("-----BEGIN CERTIFICATE-----") > 1


def test_repair_is_cached_per_host(tmp_path, monkeypatch):
    calls = []

    def _peer_certificate(host, port=443):
        calls.append(host)
        return DER_WITH_AIA

    class _Response:
        content = PEM_TEXT.encode("ascii")

        def raise_for_status(self):
            pass

    monkeypatch.setattr("eval.tls_chain.peer_certificate", _peer_certificate)
    monkeypatch.setattr("eval.tls_chain.requests.get", lambda url, timeout=None: _Response())

    repair = ChainRepair(cache_dir=tmp_path)
    assert repair.repair("example.gov.tr") is True
    assert repair.repair("example.gov.tr") is True
    assert calls == ["example.gov.tr"]  # ikinci çağrı önbellekten geldi


def test_verify_for_returns_true_until_repaired(tmp_path, monkeypatch):
    repair = ChainRepair(cache_dir=tmp_path)
    assert repair.verify_for("example.gov.tr") is True

    monkeypatch.setattr("eval.tls_chain.peer_certificate", lambda host, port=443: DER_WITH_AIA)

    class _Response:
        content = PEM_TEXT.encode("ascii")

        def raise_for_status(self):
            pass

    monkeypatch.setattr("eval.tls_chain.requests.get", lambda url, timeout=None: _Response())
    repair.repair("example.gov.tr")

    assert repair.verify_for("example.gov.tr").endswith("example.gov.tr.pem")


def test_repair_fails_gracefully_without_an_aia(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.tls_chain.peer_certificate", lambda host, port=443: b"no aia here")
    repair = ChainRepair(cache_dir=tmp_path)

    assert repair.repair("example.gov.tr") is False
    assert repair.verify_for("example.gov.tr") is True


def test_repair_fails_gracefully_when_the_handshake_fails(tmp_path, monkeypatch):
    def _boom(host, port=443):
        raise OSError("connection refused")

    monkeypatch.setattr("eval.tls_chain.peer_certificate", _boom)
    repair = ChainRepair(cache_dir=tmp_path)

    assert repair.repair("example.gov.tr") is False
