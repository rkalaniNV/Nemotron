# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace-local Ed25519 keys, created privately and never silently rotated."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def workspace_key_pair(workspace: Path, purpose: str) -> tuple[Path, Path]:
    """Create or verify a complete key pair without following trust-directory symlinks."""
    if purpose not in {"source-certification", "release-seal"}:
        raise ValueError("unsupported workspace key purpose")
    root = workspace.resolve() / ".trust"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    private_name, public_name = f"{purpose}-private.pem", f"{purpose}-public.pem"
    try:
        if stat.S_IMODE(os.fstat(directory).st_mode) & 0o022:
            raise ValueError("workspace trust directory must not be writable by other users")

        def read(name: str, *, private: bool = False) -> bytes | None:
            try:
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            except FileNotFoundError:
                return None
            with os.fdopen(descriptor, "rb") as stream:
                status = os.fstat(stream.fileno())
                if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                    raise ValueError(f"workspace key {name} must be a regular file with one link")
                if stat.S_IMODE(status.st_mode) & 0o022:
                    raise ValueError(f"workspace key {name} must not be writable by other users")
                if private and stat.S_IMODE(status.st_mode) != 0o600:
                    raise ValueError(f"workspace private key {name} must have permissions 0600")
                return stream.read()

        private_bytes, public_bytes = read(private_name, private=True), read(public_name)
        if (private_bytes is None) != (public_bytes is None):
            raise ValueError("workspace key pair is incomplete; restore the original pair")
        if private_bytes is None:
            key = Ed25519PrivateKey.generate()
            private_bytes = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            public_bytes = key.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
            for name, payload in ((private_name, private_bytes), (public_name, public_bytes)):
                descriptor = os.open(
                    name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600, dir_fd=directory,
                )
                with os.fdopen(descriptor, "wb") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
        else:
            key = serialization.load_pem_private_key(private_bytes, password=None)
            public = serialization.load_pem_public_key(public_bytes)
            if not isinstance(key, Ed25519PrivateKey) or not isinstance(public, Ed25519PublicKey):
                raise ValueError("workspace keys must use Ed25519")
            if key.public_key().public_bytes_raw() != public.public_bytes_raw():
                raise ValueError("workspace key pair does not match; restore the original pair")
    finally:
        os.close(directory)
    return root / private_name, root / public_name
