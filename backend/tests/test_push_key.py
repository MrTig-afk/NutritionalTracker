"""Web push signing key: pywebpush only reads raw or DER base64 from a string, never PEM, so the server must
load the key itself. Every push failed silently before this (found 2026-09-25).

Run:  venv/Scripts/python -m unittest backend.tests.test_push_key -v
No network: webpush is patched.
"""
import base64
import unittest
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from py_vapid import Vapid

import main


def _key():
    k = ec.generate_private_key(ec.SECP256R1())
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption()).decode()
    pub = k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    raw = k.private_numbers().private_value.to_bytes(32, "big")
    return k, pem, pub, base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _public(v: Vapid) -> bytes:
    return v.public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


class LoadKey(unittest.TestCase):
    def test_a_pem_key_loads(self):
        k, pem, _, _ = _key()
        v = main._load_vapid(pem)
        self.assertIsInstance(v, Vapid)
        self.assertEqual(_public(v), k.public_key().public_bytes(serialization.Encoding.X962,
                                                                 serialization.PublicFormat.UncompressedPoint))

    def test_a_pem_with_escaped_newlines_loads(self):   # how a one-line env var carries it
        _, pem, _, _ = _key()
        self.assertIsInstance(main._load_vapid(pem.replace("\n", "\\n")), Vapid)

    def test_the_first_private_key_wins_in_a_pasted_block(self):
        # What Render held: private key, public key, a stray line, then a second, broken private key.
        k, pem, pub, _ = _key()
        pasted = pem + "\n" + pub + "\nBKq3" + "x" * 83 + "\n\n\n" + "-----BEGIN PRIVATE KEY-----\nnot-a-key\n-----END PRIVATE KEY-----\n"
        v = main._load_vapid(pasted)
        self.assertEqual(_public(v), k.public_key().public_bytes(serialization.Encoding.X962,
                                                                 serialization.PublicFormat.UncompressedPoint))

    def test_a_raw_base64_key_loads(self):
        _, _, _, raw = _key()
        self.assertIsInstance(main._load_vapid(raw), Vapid)

    def test_missing_or_garbage_is_none(self):
        self.assertIsNone(main._load_vapid(""))
        self.assertIsNone(main._load_vapid("your-vapid-private-key"))


class Send(unittest.TestCase):
    def test_pushes_are_signed_with_the_loaded_key_object(self):
        _, pem, _, _ = _key()
        key = main._load_vapid(pem)
        with mock.patch.object(main, "VAPID_KEY", key), mock.patch.object(main, "VAPID_PUBLIC_KEY", "pub"), \
                mock.patch.object(main, "webpush") as wp:
            main._webpush_all([{"endpoint": "https://web.push.apple.com/x", "keys": {}}], "T", "B")
        self.assertIs(wp.call_args.kwargs["vapid_private_key"], key)

    def test_a_failed_push_is_logged_where_render_shows_it(self):
        _, pem, _, _ = _key()
        with mock.patch.object(main, "VAPID_KEY", main._load_vapid(pem)), mock.patch.object(main, "VAPID_PUBLIC_KEY", "pub"), \
                mock.patch.object(main, "webpush", side_effect=RuntimeError("boom")), \
                self.assertLogs(main.logger, level="WARNING") as logs:
            main._webpush_all([{"endpoint": "https://web.push.apple.com/secret-path", "keys": {}}], "T", "B")
        self.assertIn("web.push.apple.com", logs.output[0])
        self.assertNotIn("secret-path", logs.output[0])   # the endpoint is a credential: host only
