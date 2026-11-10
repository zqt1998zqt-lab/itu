from __future__ import annotations

import hashlib
from io import BytesIO
import json
import stat
import zipfile

import numpy as np
import pytest
import soundfile as sf

from server.acoustic import service
from server.acoustic.acceptance_contract import (
    contract_items,
    required_conditions,
)
from server.acoustic.acceptance_dataset import (
    AcceptanceDatasetError,
    dataset_asset,
    get_dataset,
    import_dataset,
    list_datasets,
    template_manifest,
)


SIGNALS = ("microphone", "reference", "far_end", "noise")


def _wav(
    values: np.ndarray,
    rate: int = 16_000,
    subtype: str = "PCM_16",
) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, values, rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


def _zip(files: list[tuple[str, bytes]]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files:
            archive.writestr(name, payload)
    return buffer.getvalue()


def _manifest_bytes(manifest: dict) -> bytes:
    return json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _shared_signals() -> dict[str, bytes]:
    frames = 16_000
    time = np.arange(frames) / 16_000
    mono = 0.1 * np.sin(2 * np.pi * 440 * time)
    return {
        "audio/shared/microphone.wav": _wav(np.column_stack((mono, mono * 0.5))),
        "audio/shared/reference.wav": _wav(mono),
        "audio/shared/far_end.wav": _wav(mono * 0.25),
        "audio/shared/noise.wav": _wav(
            np.zeros(24_000),
        ),
    }


def _shared_manifest() -> dict:
    manifest = template_manifest()
    for case in manifest["cases"]:
        case["signals"] = {
            role: f"audio/shared/{role}.wav" for role in SIGNALS
        }
        case["analysis_start_s"] = 0.1
    return manifest


def _valid_archive(
    manifest: dict | None = None,
    *,
    signals: dict[str, bytes] | None = None,
) -> bytes:
    manifest = _shared_manifest() if manifest is None else manifest
    signals = _shared_signals() if signals is None else signals
    return _zip(
        [("manifest.json", _manifest_bytes(manifest)), *signals.items()]
    )


def test_template_groups_the_complete_336_item_contract():
    manifest = template_manifest()
    assert manifest["schema_version"] == 1
    assert isinstance(manifest["name"], str) and manifest["name"]
    rows = contract_items()
    rows_by_id = {row["id"]: row for row in rows}
    covered = [
        item_id
        for case in manifest["cases"]
        for item_id in case["item_ids"]
    ]
    assert len(covered) == 336
    assert len(set(covered)) == 336
    assert set(covered) == set(rows_by_id)

    condition_groups = set()
    for case in manifest["cases"]:
        condition_key = tuple(case["conditions"].items())
        assert condition_key not in condition_groups
        condition_groups.add(condition_key)
        assert set(case["conditions"]) == {
            "mode",
            "direction",
            "scene",
            "volume",
            "window_ms",
            "input_level_db",
        }
        assert set(case["signals"]) == set(SIGNALS)
        for role, relative in case["signals"].items():
            assert relative == f"audio/{case['id']}/{role}.wav"
        for item_id in case["item_ids"]:
            assert required_conditions(rows_by_id[item_id]) == case["conditions"]


def test_valid_import_get_list_and_asset_resolution(tmp_path, monkeypatch):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)

    imported = import_dataset(_valid_archive())
    assert set(imported) == {
        "dataset_id",
        "name",
        "case_count",
        "item_count",
        "manifest_sha256",
        "created_at",
    }
    assert imported["item_count"] == 336
    assert imported["case_count"] == len(template_manifest()["cases"])

    stored = get_dataset(imported["dataset_id"])
    assert stored is not None
    assert stored["manifest"] == _shared_manifest()
    assert set(stored["assets"]) == set(_shared_signals())
    for metadata in stored["assets"].values():
        assert set(metadata) == {
            "sha256",
            "sample_rate",
            "frames",
            "channels",
            "duration_s",
        }
    assert stored["assets"]["audio/shared/microphone.wav"]["channels"] == 2
    assert stored["assets"]["audio/shared/reference.wav"]["channels"] == 1
    assert stored["assets"]["audio/shared/noise.wav"]["duration_s"] == 1.5
    assert stored["integrity"]["complete"] is True
    assert stored["integrity"]["covered_item_count"] == 336

    manifest_path = (
        build
        / "acceptance-datasets"
        / imported["dataset_id"]
        / "manifest.json"
    )
    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        == imported["manifest_sha256"]
    )
    summary_path = manifest_path.with_name("summary.json")
    assets_path = manifest_path.with_name("assets.json")
    summary = json.loads(summary_path.read_text())
    asset_index = json.loads(assets_path.read_text())
    assert assets_path.read_bytes() == _manifest_bytes(asset_index)
    assert (
        summary["assets_sha256"]
        == hashlib.sha256(_manifest_bytes(asset_index)).hexdigest()
    )

    listed = list_datasets()
    assert len(listed) == 1
    assert listed[0]["dataset_id"] == imported["dataset_id"]
    assert "manifest" not in listed[0]
    assert "assets" not in listed[0]
    assert listed[0]["asset_count"] == 4

    relative = "audio/shared/microphone.wav"
    resolved = dataset_asset(imported["dataset_id"], relative)
    assert resolved.is_file()
    assert resolved.is_relative_to(
        build / "acceptance-datasets" / imported["dataset_id"] / "assets"
    )
    assert (
        hashlib.sha256(resolved.read_bytes()).hexdigest()
        == stored["assets"][relative]["sha256"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("condition", "conditions do not match"),
        ("duplicate", "contain duplicates"),
        ("missing", "missing contract item_ids"),
        ("unknown", "unknown ids"),
    ],
)
def test_manifest_contract_errors_leave_no_dataset(
    tmp_path, monkeypatch, mutation, message
):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    manifest = _shared_manifest()
    if mutation == "condition":
        manifest["cases"][0]["conditions"]["direction"] = "wrong"
    elif mutation == "duplicate":
        manifest["cases"][0]["item_ids"].append(
            manifest["cases"][0]["item_ids"][0]
        )
    elif mutation == "missing":
        case = next(case for case in manifest["cases"] if len(case["item_ids"]) > 1)
        case["item_ids"].pop()
    else:
        manifest["cases"][0]["item_ids"].append("XX-unknown:Z99")

    with pytest.raises(AcceptanceDatasetError, match=message):
        import_dataset(_valid_archive(manifest))
    assert list_datasets() == []
    store = build / "acceptance-datasets"
    preserved = [
        path for path in store.iterdir() if path.name.startswith(".import-")
    ]
    assert len(preserved) == 1
    assert preserved[0].is_dir()


@pytest.mark.parametrize(
    "case_id",
    [
        "../escape",
        "/absolute",
        "nested/case",
        "case.with.dot",
        "case id",
        "a" * 129,
    ],
)
def test_manifest_rejects_unsafe_case_ids(
    tmp_path, monkeypatch, case_id
):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    manifest = _shared_manifest()
    manifest["cases"][0]["id"] = case_id

    with pytest.raises(
        AcceptanceDatasetError,
        match=r"manifest\.cases\[0\]\.id must match",
    ):
        import_dataset(_valid_archive(manifest))
    assert list_datasets() == []
    store = build / "acceptance-datasets"
    assert len(
        [path for path in store.iterdir() if path.name.startswith(".import-")]
    ) == 1


@pytest.mark.parametrize(
    "unsafe_name",
    ["../manifest.json", "/manifest.json", "audio\\microphone.wav"],
)
def test_zip_rejects_traversal_absolute_and_backslash_names(
    tmp_path, monkeypatch, unsafe_name
):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    payload = _zip(
        [
            ("manifest.json", _manifest_bytes(_shared_manifest())),
            (unsafe_name, b"not audio"),
        ]
    )
    with pytest.raises(
        AcceptanceDatasetError,
        match="relative|backslash|absolute|parent",
    ):
        import_dataset(payload)


def test_zip_rejects_duplicate_names_and_symlink_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    manifest = _manifest_bytes(_shared_manifest())
    duplicate = _zip(
        [("manifest.json", manifest), ("manifest.json", manifest)]
    )
    with pytest.raises(AcceptanceDatasetError, match="duplicate ZIP member"):
        import_dataset(duplicate)

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", manifest)
        link = zipfile.ZipInfo("audio/shared/microphone.wav")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "outside.wav")
    with pytest.raises(AcceptanceDatasetError, match="symlink"):
        import_dataset(buffer.getvalue())


def test_nonfinite_audio_and_mismatched_timeline_are_rejected(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    signals = _shared_signals()
    invalid = np.zeros((16_000, 2))
    invalid[10, 0] = np.nan
    signals["audio/shared/microphone.wav"] = _wav(invalid, subtype="FLOAT")
    with pytest.raises(AcceptanceDatasetError, match="non-finite"):
        import_dataset(_valid_archive(signals=signals))

    signals = _shared_signals()
    signals["audio/shared/far_end.wav"] = _wav(np.zeros(16_000), rate=8_000)
    with pytest.raises(AcceptanceDatasetError, match="sample rates do not match"):
        import_dataset(_valid_archive(signals=signals))

    signals = _shared_signals()
    signals["audio/shared/reference.wav"] = _wav(np.zeros(15_999))
    with pytest.raises(AcceptanceDatasetError, match="timelines differ"):
        import_dataset(_valid_archive(signals=signals))


def test_asset_lookup_rejects_traversal_and_replaced_symlink(
    tmp_path, monkeypatch
):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    imported = import_dataset(_valid_archive())
    with pytest.raises(AcceptanceDatasetError, match="relative|parent|absolute"):
        dataset_asset(imported["dataset_id"], "../outside.wav")

    relative = "audio/shared/microphone.wav"
    asset = dataset_asset(imported["dataset_id"], relative)
    preserved_original = asset.with_name("microphone.original.wav")
    asset.rename(preserved_original)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(_wav(np.zeros(16_000)))
    asset.symlink_to(outside)
    assert preserved_original.is_file()
    with pytest.raises(AcceptanceDatasetError, match="symlink"):
        dataset_asset(imported["dataset_id"], relative)


def test_persisted_manifest_hash_and_contract_tampering_are_rejected(
    tmp_path, monkeypatch
):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    imported = import_dataset(_valid_archive())
    dataset_root = build / "acceptance-datasets" / imported["dataset_id"]
    manifest_path = dataset_root / "manifest.json"
    summary_path = dataset_root / "summary.json"
    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        == imported["manifest_sha256"]
    )

    manifest_path.write_text('{"schema_version":1,"name":"tampered"}')
    assert get_dataset(imported["dataset_id"]) is None
    assert list_datasets() == []

    invalid_manifest = {
        "schema_version": 1,
        "name": "tampered",
        "cases": [],
    }
    manifest_blob = _manifest_bytes(invalid_manifest)
    manifest_path.write_bytes(manifest_blob)
    summary = json.loads(summary_path.read_text())
    summary["manifest_sha256"] = hashlib.sha256(manifest_blob).hexdigest()
    summary_path.write_text(json.dumps(summary, sort_keys=True))

    assert get_dataset(imported["dataset_id"]) is None
    assert list_datasets() == []
    with pytest.raises(AcceptanceDatasetError, match="integrity"):
        dataset_asset(
            imported["dataset_id"],
            "audio/shared/microphone.wav",
        )


def test_persisted_assets_index_tampering_is_rejected(
    tmp_path, monkeypatch
):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    imported = import_dataset(_valid_archive())
    dataset_root = build / "acceptance-datasets" / imported["dataset_id"]
    assets_path = dataset_root / "assets.json"
    assets = json.loads(assets_path.read_text())
    assets["audio/shared/microphone.wav"]["sha256"] = "0" * 64
    assets_path.write_bytes(_manifest_bytes(assets))

    assert get_dataset(imported["dataset_id"]) is None
    assert list_datasets() == []
    with pytest.raises(AcceptanceDatasetError, match="integrity"):
        dataset_asset(
            imported["dataset_id"],
            "audio/shared/microphone.wav",
        )


def test_legacy_summary_without_assets_hash_requires_reimport(
    tmp_path, monkeypatch
):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    imported = import_dataset(_valid_archive())
    dataset_root = build / "acceptance-datasets" / imported["dataset_id"]
    summary_path = dataset_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.pop("assets_sha256")
    summary_path.write_text(json.dumps(summary, sort_keys=True))
    preserved_summary = summary_path.read_bytes()

    with pytest.raises(AcceptanceDatasetError, match="re-import"):
        get_dataset(imported["dataset_id"])
    assert list_datasets() == []
    with pytest.raises(AcceptanceDatasetError, match="re-import"):
        dataset_asset(
            imported["dataset_id"],
            "audio/shared/microphone.wav",
        )
    assert summary_path.read_bytes() == preserved_summary
