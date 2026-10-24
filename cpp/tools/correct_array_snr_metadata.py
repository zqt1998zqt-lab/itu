#!/usr/bin/env python3
"""Create an immutable array-manifest version using measured quantized SNR."""
import argparse
import copy
import json
from pathlib import Path

import audit_dataset_quality_deep as audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve();binding=audit.digest(source)
    manifest=json.loads(source.read_text());updated=copy.deepcopy(manifest)
    corrections=[];checked=[]
    for scene in updated["scenes"]:
        folder=Path(scene.get("source_dataset",source.parent))/"cases"/scene.get("source_case_id",scene["id"])
        scene["source_dataset"]=str(folder.parents[1])
        scene["source_case_id"]=folder.name
        row=audit.check_scene(("array_metadata_v2",folder,scene))
        if row["errors"] and row["errors"]!=["measured_snr_metadata_error_over_0.001_db"]:
            raise ValueError(f"unrelated input problem: {scene['id']} {row['errors']}")
        if row["errors"]:
            before=scene["measured_input_snr_db"]
            scene["parent_measured_input_snr_db"]=before
            scene["measured_input_snr_db"]=row["scheduled_snr_db"]
            corrections.append(dict(id=scene["id"],before=before,after=scene["measured_input_snr_db"],
                                    change_db=scene["measured_input_snr_db"]-before))
        row=audit.check_scene(("array_metadata_v2",folder,scene))
        if row["errors"]:
            raise ValueError(f"correction did not resolve audit: {row}")
        checked.append(row)
    updated["parent_manifest"]=str(source)
    updated["parent_manifest_sha256"]=binding
    updated["metadata_corrections"]=corrections
    updated["scope"]="quantized-component SNR metadata correction only; original PCM hashes and nominal grid unchanged"
    if audit.digest(source)!=binding:
        raise ValueError("parent changed")
    args.destination.parent.mkdir(parents=True,exist_ok=False)
    args.destination.write_text(json.dumps(updated,indent=2)+"\n")
    report=dict(scenes=len(checked),scenes_with_errors=0,corrections=corrections,parent_sha256=binding,
                manifest_sha256=audit.digest(args.destination),pcm_modified=False,parent_unchanged=True)
    (args.destination.parent/"correction-report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":main()
