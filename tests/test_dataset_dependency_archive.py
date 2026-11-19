from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from server.acoustic.dataset_dependency_archive import (
    DatasetArchiveError, DatasetMetadataArchive, SCHEMA,
    decode_metadata, encode_metadata,
)

IDENTIFIER = "a" * 32


def _write(path, value):
    payload = json.dumps(value, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _fixture(root, *, relative=None, wrong_id=False):
    record = {"dataset_id": IDENTIFIER, "kind": "scene", "execution_allowed": False,
              "dataset": {"dataset_id": "b" * 32 if wrong_id else IDENTIFIER,
                          "root": Path("/not-readable/legacy"), "referenced_assets": {"audio/one.wav"},
                          "contract_item_ids": ("WB-wired:B24",), "cases": [{"case_id": "case-1"}]},
              "contracts": {"WB-wired:B24": {"id": "WB-wired:B24"}}}
    relative = relative or f"datasets/{IDENTIFIER}.json"
    digest, size = _write(root / f"datasets/{IDENTIFIER}.json", encode_metadata(record))
    manifest = {"schema": SCHEMA, "datasets": {IDENTIFIER: {"kind": "scene", "path": relative,
                                                            "sha256": digest, "bytes": size}}}
    manifest_sha, _ = _write(root / "manifest.json", manifest)
    _write(root / "retirement.json", {"schema": "itu-dataset-retirement-v1", "retired_dataset_ids": [IDENTIFIER],
                                      "metadata_archive_manifest_sha256": manifest_sha})
    return root


def test_metadata_roundtrip_preserves_types_and_has_no_audio_io(tmp_path):
    archive = DatasetMetadataArchive(_fixture(tmp_path))
    value = archive.load(IDENTIFIER)
    assert value["dataset"]["root"] == Path("/not-readable/legacy")
    assert value["dataset"]["referenced_assets"] == {"audio/one.wav"}
    assert value["dataset"]["contract_item_ids"] == ("WB-wired:B24",)
    assert value["execution_allowed"] is False
    assert decode_metadata(encode_metadata(value)) == value


def test_history_callers_get_independent_mutable_values(tmp_path):
    archive = DatasetMetadataArchive(_fixture(tmp_path))
    first = archive.load(IDENTIFIER)
    first["dataset"]["cases"].clear()
    assert len(archive.load(IDENTIFIER)["dataset"]["cases"]) == 1


def test_manifest_tampering_is_rejected(tmp_path):
    root = _fixture(tmp_path)
    with (root / "manifest.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(DatasetArchiveError, match="manifest SHA"):
        DatasetMetadataArchive(root)


def test_dataset_tampering_is_rejected_even_after_reader_created(tmp_path):
    root = _fixture(tmp_path)
    archive = DatasetMetadataArchive(root)
    archive.load(IDENTIFIER)
    with (root / f"datasets/{IDENTIFIER}.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(DatasetArchiveError, match="SHA/size"):
        archive.load(IDENTIFIER)


def test_parent_traversal_and_unknown_dataset_are_rejected(tmp_path):
    archive = DatasetMetadataArchive(_fixture(tmp_path, relative="../stolen.json"))
    with pytest.raises(DatasetArchiveError, match="unsafe"):
        archive.load(IDENTIFIER)
    with pytest.raises(DatasetArchiveError, match="not archived"):
        archive.load("b" * 32)


def test_dataset_identity_mismatch_is_rejected(tmp_path):
    archive = DatasetMetadataArchive(_fixture(tmp_path, wrong_id=True))
    with pytest.raises(DatasetArchiveError, match="identity"):
        archive.load(IDENTIFIER)


def test_symlink_root_and_missing_archive_fail_closed(tmp_path):
    root = _fixture(tmp_path / "source")
    link = tmp_path / "linked"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(DatasetArchiveError, match="symlink"):
        DatasetMetadataArchive(link)
    with pytest.raises(DatasetArchiveError, match="regular file"):
        DatasetMetadataArchive(tmp_path / "missing")


@pytest.mark.parametrize("scene,mode", [("sce4", "SWB-wireless"), ("sce3", "NB-wired")])
def test_real_history_reader_validates_positional_selection(scene, mode):
    from server.app import app
    from server.acoustic import scene_condition_batch
    assert app is not None
    identifier = "008e00742ec6c283114e28fe386a9cd1"
    with pytest.raises((ValueError, TypeError)):
        scene_condition_batch._validated_dataset(identifier, scene, mode)
