#!/usr/bin/env python3
"""Separate speech defects from residual transient noise without changing PASS."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def classify(row):
    if row["passed"] != (not row["failures"]):
        raise ValueError("PASS and failure evidence disagree")
    if row["passed"]:
        return "strict_pass"
    if not row["failures"]:
        raise ValueError("failed row has no failure evidence")
    if any(not f.endswith(":attenuation") for f in row["failures"]):
        return "speech_or_signal_integrity"
    if row["noise_class"] == "transient":
        return "transient_attenuation_only"
    if row["noise_class"] == "steady":
        return "steady_attenuation_only"
    raise ValueError("missing temporal noise classification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    details, summary, bindings = [], {}, {}
    for path in args.results:
        data = path.read_bytes()
        bindings[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        groups = collections.defaultdict(list)
        for line in data.decode().splitlines():
            row = json.loads(line)
            if row.get("treatment", "off") != "off":
                continue
            category = classify(row)
            profile = row.get("profile", row["variant"])
            groups[profile].append((row, category))
            details.append(dict(source=str(path.resolve()), profile=profile,
                run=row["run"], id=row["id"], noise=row["noise"], noise_class=row["noise_class"],
                snr_db=row["snr_db"], strict_passed=row["passed"], failures=row["failures"],
                category=category, action="review_transient_and_background" if category == "transient_attenuation_only"
                       else "repair" if category != "strict_pass" else "retain"))
        for profile, group in groups.items():
            keys = {(r["run"], r["id"]) for r, _ in group}
            if len(keys) != len(group):
                raise ValueError("duplicate cases in acceptance profile")
            summary[str(path.resolve()) + "::" + profile] = dict(total=len(group),
                strict_passed=sum(r["passed"] for r, _ in group),
                categories=dict(collections.Counter(c for _, c in group)))
    result = dict(thresholds_changed=False, original_pass_fail_preserved=True,
        transient_residual_is_not_certified_pass=True,
        transient_label_does_not_exempt_sustained_background=True,
        classifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), summary=summary, cases=details,
        source_sha256=bindings)
    (out / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
