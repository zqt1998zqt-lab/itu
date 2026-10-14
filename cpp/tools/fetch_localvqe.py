#!/usr/bin/env python3
"""Fetch pinned optional LocalVQE/GGML sources and allowlisted F32 weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

REVISION = "f53063c9eb2a85f96479867d1dd911dc3bf6319b"
GGML_REVISION = "c044a8eeae2591faa0950c8b5e514cbc4bbfc4ca"
SOURCES = (
    (f"localvqe-{REVISION}", f"https://codeload.github.com/localai-org/LocalVQE/tar.gz/{REVISION}",
     "2b2cc18ddfcf1d5b49f503ab0f066ea045a6624f212be96535a936476cd62a8b"),
    (f"ggml-{GGML_REVISION}", f"https://codeload.github.com/ggml-org/ggml/tar.gz/{GGML_REVISION}",
     "d5bdf6f42a60368ee6a73076492a880d66f9e4435bffc82992ea6dd28b70bb0d"),
)
MODELS = (
    ("localvqe-v1.2-1.3M-f32.gguf", "4856ecf5f522b23fb2bc5caeac81f323c0ef1c4c156a9c7d40a6adbe092ba9ce"),
    ("localvqe-v1.4-aec-2.7K-f32.gguf", "d79f824f6ee6f58b2a7108d1df90db51c42c15a7400a5addbd3be9972ba657a4"),
    ("localvqe-pi-v1-49k-f32.gguf", "0e0c82a8e9703e818b64dedd0fc306394cf5bbb59fcec1ccca82099d352d0c26"),
)


def cached_download(path, url, expected, offline):
    if path.exists():
        data = path.read_bytes()
    elif offline:
        raise ValueError(f"missing offline artifact: {path}")
    else:
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}; refusing to use artifact")
    if not path.exists():
        path.write_bytes(data)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=Path("build/deps"))
    p.add_argument("--offline", action="store_true", help="Verify/reuse cached archives and models only")
    a = p.parse_args()
    root = a.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url, sha in SOURCES:
        data = cached_download(root / f"{name}.tar.gz", url, sha, a.offline)
        source = root / name
        if not source.exists():
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                top = {Path(m.name).parts[0] for m in archive.getmembers()}
                if len(top) != 1:
                    raise ValueError("unexpected source archive layout")
                unpacked = root / next(iter(top))
                if unpacked.exists():
                    raise ValueError("extraction would overwrite an existing directory")
                archive.extractall(root, filter="data")
                unpacked.rename(source)
        records.append(dict(path=name, url=url, archive_sha256=sha))
    localvqe, ggml = root / SOURCES[0][0], root / SOURCES[1][0]
    vendor = localvqe / "ggml/vendor/ggml"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    if vendor.is_symlink():
        if vendor.resolve() != ggml:
            raise ValueError("GGML symlink points to a different revision")
    elif vendor.exists():
        if any(vendor.iterdir()):
            raise ValueError("refusing to replace a populated GGML checkout")
        vendor.rmdir()
        vendor.symlink_to(ggml, target_is_directory=True)
    else:
        vendor.symlink_to(ggml, target_is_directory=True)
    models = root / "localvqe-models"
    models.mkdir(exist_ok=True)
    for filename, sha in MODELS:
        url = f"https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/{filename}"
        cached_download(models / filename, url, sha, a.offline)
        records.append(dict(path=str((models / filename).relative_to(root)), url=url, sha256=sha))
    lock = dict(localvqe_revision=REVISION, ggml_revision=GGML_REVISION, artifacts=records,
                ggml_patch="ggml/patches/ggml-gru.patch, applied by upstream CMake; extracted source may be patched",
                license="Apache-2.0 LocalVQE; MIT GGML", optional=True)
    (root / "ITU_LOCALVQE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(dict(source=str(localvqe), models=str(models), lock=str(root / "ITU_LOCALVQE_LOCK.json")), indent=2))


if __name__ == "__main__":
    main()
