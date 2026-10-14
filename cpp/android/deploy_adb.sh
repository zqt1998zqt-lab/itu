#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "$0")/../.." && pwd)
build_dir=${1:-"$project_root/build/android-arm64"}
serial=${2:-${ANDROID_SERIAL:-cb8145db}}
[[ "$serial" =~ ^[a-zA-Z0-9._:-]+$ ]] || { echo 'Invalid adb serial' >&2; exit 1; }
adb -s "$serial" get-state
abi=$(adb -s "$serial" shell getprop ro.product.cpu.abi | tr -d '\r')
[ "$abi" = arm64-v8a ] || { echo "Expected arm64-v8a, got $abi" >&2; exit 1; }
run_id="run-$(date -u +%Y%m%dT%H%M%SZ)"
device_dir="/data/local/tmp/itu_audio_cpp/$run_id"
reports="$project_root/build/device/$run_id"
bundle="$reports/bundle"
mkdir -p "$bundle" "$reports"
for item in itu_audio_tests itu_audio_bench itu_audio_cli libitu_audio_native.so; do
  cp "$build_dir/cpp/$item" "$bundle/$item"
done
if [ -f "$project_root/build/apk/itu-audio-debug.apk" ]; then cp "$project_root/build/apk/itu-audio-debug.apk" "$bundle/"; fi
python3 "$project_root/cpp/tools/make_fixture.py" "$bundle/fixture"
(cd "$bundle" && sha256sum itu_audio_tests itu_audio_bench itu_audio_cli libitu_audio_native.so fixture/*.wav fixture/source.json > SHA256SUMS)
if [ -f "$bundle/itu-audio-debug.apk" ]; then (cd "$bundle" && sha256sum itu-audio-debug.apk >> SHA256SUMS); fi
adb -s "$serial" shell mkdir -p "$device_dir"
adb -s "$serial" push "$bundle/." "$device_dir/"
adb -s "$serial" shell "cd $device_dir && sha256sum -c SHA256SUMS" | tee "$reports/hash-check.txt"
adb -s "$serial" shell "chmod 700 $device_dir/itu_audio_tests $device_dir/itu_audio_bench $device_dir/itu_audio_cli"
adb -s "$serial" shell "cd $device_dir && ./itu_audio_tests" | tee "$reports/tests.txt"
adb -s "$serial" shell "cd $device_dir && ./itu_audio_bench --frames 2000 --channels 1" > "$reports/bench-mono.json"
adb -s "$serial" shell "cd $device_dir && ./itu_audio_bench --frames 2000 --channels 4" > "$reports/bench-4mic.json"
adb -s "$serial" shell "cd $device_dir && ./itu_audio_cli --mic fixture/before.wav --echo fixture/echo.wav --beam-delays 3,0 --far-start 3 --far-end 4 --out result"
adb -s "$serial" pull "$device_dir/result.json" "$reports/"
for suffix in after.wav aec.wav linear-aec.wav bf.wav frames.csv; do adb -s "$serial" pull "$device_dir/result.$suffix" "$reports/"; done
adb -s "$serial" shell getprop ro.product.model > "$reports/model.txt"
adb -s "$serial" shell getprop ro.build.version.sdk > "$reports/sdk.txt"
adb -s "$serial" shell getconf PAGESIZE > "$reports/page-size.txt"
printf '%s\n' "$serial" > "$reports/serial.txt"
printf '%s\n' "$device_dir" > "$reports/device-directory.txt"
if [ -f "$bundle/itu-audio-debug.apk" ]; then
  adb -s "$serial" install -r "$bundle/itu-audio-debug.apk"
  adb -s "$serial" shell run-as dev.itu.audio rm -f files/selftest.json
  adb -s "$serial" shell am start -S -n dev.itu.audio/.MainActivity --ez self_test true --es run_id "$run_id"
  found=0
  for attempt in $(seq 1 30); do
    adb -s "$serial" exec-out run-as dev.itu.audio cat files/selftest.json > "$reports/jni-selftest.json.tmp" 2>/dev/null || true
    if python3 - "$reports/jni-selftest.json.tmp" "$run_id" <<'PY'
import json,sys
try:
    result=json.load(open(sys.argv[1]))
    assert result.get('passed') is True and result.get('checks',0)>=10 and result.get('run_id')==sys.argv[2]
except (OSError,ValueError,AssertionError):
    sys.exit(1)
PY
    then
      mv "$reports/jni-selftest.json.tmp" "$reports/jni-selftest.json"; found=1; break
    fi
    sleep 1
  done
  [ "$found" = 1 ] || { echo 'JNI report missing; inspect app log' >&2; exit 1; }
fi
python3 - "$reports" "$run_id" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
for name in ['bench-mono.json','bench-4mic.json','result.json']:
    obj=json.loads((p/name).read_text())
    print(name, obj)
if (p/'jni-selftest.json').exists():
    obj=json.loads((p/'jni-selftest.json').read_text())
    assert obj['passed'] and obj['checks'] >= 10 and obj['run_id']==sys.argv[2], obj
    print('jni-selftest.json', obj)
PY
echo "$reports"
