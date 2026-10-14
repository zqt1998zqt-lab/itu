#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "$0")/../.." && pwd)
serial=${1:-${ANDROID_SERIAL:-cb8145db}}
destination=${2:-"$project_root/build/capture/$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$destination"
for item in capture.before.wav capture.after.wav capture.json; do
  adb -s "$serial" exec-out run-as dev.itu.audio cat "files/$item" > "$destination/$item"
done
python3 - "$destination" <<'PY'
import json, pathlib, sys, wave
p=pathlib.Path(sys.argv[1]); report=json.loads((p/'capture.json').read_text())
assert report['input_kind']=='device microphone PCM16' and report['frames']>0
for name in ['capture.before.wav','capture.after.wav']:
    with wave.open(str(p/name)) as w:
        assert w.getframerate()==16000 and w.getnchannels()==1 and w.getsampwidth()==2
    assert w.getnframes()==report['frames']*report['frame_size']+(report['dsp_delay_samples'] if name.endswith('after.wav') else 0)
        print(name, w.getnframes(), 'samples')
print(report)
PY
python3 "$project_root/cpp/tools/verify_capture.py" "$destination"
echo "$destination"
