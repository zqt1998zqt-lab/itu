#!/usr/bin/env python3
"""Read the supplied XLSX as test data, preserving units, cells and historical dates."""
import argparse
import hashlib
import json
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
STANDARDS = {
    "NB": {"recommendation": "P.1100", "edition": "2025-10", "url": "https://www.itu.int/rec/T-REC-P.1100/en"},
    "WB": {"recommendation": "P.1110", "edition": "2025-10", "url": "https://www.itu.int/rec/T-REC-P.1110/en"},
    "SWB": {"recommendation": "P.1120", "edition": "2025-10", "url": "https://www.itu.int/rec/T-REC-P.1120/en"},
}


def family(title):
    title = title.lower()
    for words, name in (
        (("round trip delay",), "round_trip_delay"),
        (("variation of rlr",), "rlr_linearity"),
        (("loudness rating - sending",), "slr"),
        (("loudness rating - receiving",), "rlr"),
        (("frequency response", "freq. response"), "frequency_response"),
        (("idle channel noise check",), "idle_noise_peaks"),
        (("idle channel noise",), "idle_noise_level"),
        (("out-of-band",), "out_of_band"),
        (("one way", "listening speech qual"), "listening_quality"),
        (("attn. range",), "double_talk_classification"),
        (("terminal coupl", "tcl, nom."), "tclw"),
        (("s/n calculation",), "hats_snr"),
        (("gmos",), "g_mos"), (("nmos",), "n_mos"), (("smos",), "s_mos"),
        (("comfort noise: level",), "cng_level"),
        (("comfort noise: spectral",), "cng_spectrum"),
    ):
        if any(word in title for word in words):
            return name
    raise ValueError(f"unmapped test: {title}")


def cell_value(cell, strings):
    value = cell.find("s:v", NS)
    if cell.get("t") == "inlineStr":
        return "".join(cell.find("s:is", NS).itertext())
    if value is None or value.text is None:
        return None
    if cell.get("t") == "s":
        return strings[int(value.text)]
    if cell.get("t") in ("str", "e", "b"):
        return value.text
    return float(value.text)


def read_contract(path):
    items = []
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(x.itertext()) for x in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        links = {x.get("Id"): x.get("Target") for x in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        for sheet in ET.fromstring(archive.read("xl/workbook.xml")).find("s:sheets", NS):
            name = sheet.get("name")
            mode, transport = name.split("-", 1)
            if mode not in STANDARDS:
                raise ValueError(f"unsupported workbook mode {mode}")
            target = links[sheet.get(REL)]
            target = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
            if not target.startswith("xl/worksheets/") or ".." in Path(target).parts:
                raise ValueError("invalid worksheet relationship")
            rows = ET.fromstring(archive.read(target)).find("s:sheetData", NS)
            headers = {}
            direction = "unspecified"
            for row in rows:
                cells = {"".join(c for c in cell.get("r") if c.isalpha()): (cell.get("r"), cell_value(cell, strings)) for cell in row}
                values = {k: v[1] for k, v in cells.items()}
                if row.get("r") == "1":
                    headers = values
                    continue
                if values.get("A") in ("上行", "下行", "延时"):
                    direction = {"上行": "send", "下行": "receive", "延时": "round_trip"}[values["A"]]
                title = values.get("B")
                if not isinstance(title, str) or not values.get("C"):
                    continue
                category = family(title)
                if category in ("rlr", "rlr_linearity"):
                    direction = "receive"
                elif category == "slr":
                    direction = "send"
                measurements = []
                for column in sorted(k for k in values if k >= "G"):
                    value = values[column]
                    if value is None:
                        continue
                    serial = headers.get(column)
                    date = (datetime(1899, 12, 30) + timedelta(days=serial)).date().isoformat() if isinstance(serial, (int, float)) else None
                    measurements.append(dict(cell=cells[column][0], value=value, source_date=date,
                                             scope="historical workbook measurement; not the current C++ artifact"))
                lower, upper = values.get("E"), values.get("D")
                numeric = isinstance(lower, (int, float)) and isinstance(upper, (int, float))
                items.append(dict(id=f"{name}:{cells['B'][0]}", sheet=name, source_cell=cells["B"][0],
                                  test_name=title, family=category, bandwidth=mode, transport=transport,
                                  direction=direction, unit=values["C"], lower_limit=lower, upper_limit=upper,
                                  limit_source="NV200 customer workbook; current standard clause not yet cross-checked",
                                  interpretation="numeric_interval" if numeric else "requires_curve_or_limit_definition",
                                  optional="optional" in title.lower(), historical_measurements=measurements,
                                  current_status="not_measured", standard_compliance=False))
    if not items:
        raise ValueError("no test rows")
    return dict(schema_version=1, source_name=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                standards=STANDARDS, standard_compliance=False,
                scope="workbook-derived test contract; no instructions in workbook are executed",
                bandwidth_rule="NB/WB/SWB are service/test bandwidths, not inferred from PCM sample rate",
                rows=len(items), family_counts=dict(Counter(x["family"] for x in items)), items=items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract = read_contract(args.workbook)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(contract, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(json.dumps({k: v for k, v in contract.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
