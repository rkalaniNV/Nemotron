from __future__ import annotations

from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.authoring_release.freeze import AuthoringFreezeError, load_frozen_release
from nemotron.steps.byob.runtime.authoring_release.handoff import AuthoringHandoffError, handoff_frozen_release
from nemotron.steps.byob.runtime.authoring_release.review import load_review_packet
from nemotron.steps.byob.runtime.authoring_release.trust import TRUST_FIELDS, pack_trust, packet_trust, trust_fields
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json, write_canonical_json
from nemotron.steps.byob.runtime.release_seal import sign_release_digest
from tests.steps.byob.test_bfcl_authoring_e2e import SEAL_AUTHORITY, SEAL_VERIFICATION, _approved_release


@pytest.mark.parametrize("mode", ["dev", "release", "compliance"])
def test_trust_is_bound_to_approved_packet_and_signed_release(tmp_path: Path, mode: str) -> None:
    release, packet, _ = _approved_release(tmp_path, "local_python", trust_mode=mode)
    loaded = load_frozen_release(release.root, **SEAL_VERIFICATION)
    expected = trust_fields(mode)
    assert packet_trust(load_review_packet(packet).document) == expected
    assert {name: loaded.manifest[name] for name in TRUST_FIELDS if name in loaded.manifest} == expected
    assert pack_trust(loaded.pack_root) == expected
    assert loaded.manifest["schema_version"] == f"bfcl-authoring-frozen-release-v{4 if expected else 3}"


    if mode == "compliance":
        return
    # Even a signing-key holder cannot upgrade the mode without another review.
    manifest = dict(release.manifest)
    manifest.update(trust_fields("release" if mode == "dev" else "dev"))
    manifest.pop("signature")
    manifest.pop("manifest_digest")
    manifest["manifest_digest"] = sha256_json(manifest)
    manifest["signature"] = sign_release_digest(SEAL_AUTHORITY, manifest["manifest_digest"])
    release.root.chmod(0o755)
    write_canonical_json(manifest, release.root / "freeze_manifest.json")
    with pytest.raises(AuthoringFreezeError, match="release_trust_mismatch"):
        load_frozen_release(release.root, **SEAL_VERIFICATION)


@pytest.mark.parametrize("mutation", [None, "gold", "omit_status", "upgrade"])
def test_dev_handoff_only_accepts_explicit_unofficial_publication(tmp_path: Path, mutation: str | None) -> None:
    release, _, _ = _approved_release(tmp_path, "local_python", trust_mode="dev")
    output = tmp_path / "output"
    calls = []

    class Publisher:
        kind = "local_python"

        def validate_pack(self, root):
            return release.pack_fingerprint

        def bind_config(self, *args):
            pass

        def prepare(self, config):
            calls.append("fresh_validation")
            return tmp_path / "validation.json"

        def load_validation_report(self, path):
            return {"gold_eligible": True}

        def require_fresh_gold(self, report, fingerprint):
            assert report["gold_eligible"] is True
            assert fingerprint == release.pack_fingerprint

        def generate(self, config):
            output.mkdir()
            (output / "benchmark.parquet").write_bytes(b"test")
            (output / "benchmark_raw.parquet").write_bytes(b"test")
            manifest = {
                **trust_fields("dev"),
                "pack": {"content_hash": release.pack_fingerprint},
                "tier": "gold", "gold_eligible": False,
                "gold_ineligibility_reasons": ["unofficial_authoring_release"],
            }
            if mutation == "gold":
                manifest["gold_eligible"] = True
            elif mutation == "omit_status":
                manifest.pop("release_status")
            elif mutation == "upgrade":
                manifest["official_publishable"] = True
            write_canonical_json(manifest, output / "run_manifest.json")
            return output / "benchmark.parquet"

        def verify_publication_manifest(self, *args):
            calls.append("verify_origin")

    if mutation is not None:
        with pytest.raises(AuthoringHandoffError):
            handoff_frozen_release(release.root, tmp_path / "config", adapter=Publisher(), **SEAL_VERIFICATION)
        assert calls == ["fresh_validation"]
    else:
        result = handoff_frozen_release(release.root, tmp_path / "config", adapter=Publisher(), **SEAL_VERIFICATION)
        assert result.run_manifest["official_publishable"] is False
        assert calls == ["fresh_validation", "verify_origin"]
