#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "$0")/../.." && pwd)
native_build=${1:-"$project_root/build/android-arm64"}
sdk_root=${ANDROID_SDK_ROOT:-"$HOME/.android-sdk"}
tools="$sdk_root/build-tools/35.0.0"
jar="$sdk_root/platforms/android-35/android.jar"
apk_dir="$project_root/build/apk"
test -f "$native_build/cpp/libitu_audio_native.so"
mkdir -p "$apk_dir/classes" "$apk_dir/dex" "$apk_dir/package/lib/arm64-v8a"
javac -encoding UTF-8 -source 8 -target 8 -classpath "$jar" -d "$apk_dir/classes" "$project_root"/cpp/android/java/dev/itu/audio/*.java
shopt -s globstar nullglob
class_files=("$apk_dir"/classes/**/*.class)
[ "${#class_files[@]}" -gt 0 ]
"$tools/d8" --lib "$jar" --min-api 26 --output "$apk_dir/dex" "${class_files[@]}"
"$tools/aapt2" link -I "$jar" --manifest "$project_root/cpp/android/AndroidManifest.xml" -o "$apk_dir/unsigned.apk"
cp "$native_build/cpp/libitu_audio_native.so" "$apk_dir/package/lib/arm64-v8a/"
if [ -n "${ITU_ORT_ROOT:-}" ]; then
  # Use an Android arm64 SDK, never the Linux aarch64 library.
  cp "$ITU_ORT_ROOT/lib/libonnxruntime.so" "$apk_dir/package/lib/arm64-v8a/"
fi
cp "$apk_dir/dex/classes.dex" "$apk_dir/package/"
mkdir -p "$apk_dir/package/assets/licenses"
cp "$project_root"/cpp/licenses/* "$apk_dir/package/assets/licenses/"
(cd "$apk_dir/package" && zip -q "$apk_dir/unsigned.apk" classes.dex lib/arm64-v8a/*.so)
(cd "$apk_dir/package" && zip -qr "$apk_dir/unsigned.apk" assets/licenses)
"$tools/zipalign" -f -P 16 4 "$apk_dir/unsigned.apk" "$apk_dir/aligned.apk"
if [ ! -f "$apk_dir/debug.keystore" ]; then
  keytool -genkeypair -keystore "$apk_dir/debug.keystore" -storepass android -keypass android -alias androiddebugkey \
    -dname 'CN=ITU Phase1 Debug,O=ITU,C=CN' -keyalg RSA -keysize 2048 -validity 3650 >/dev/null
fi
"$tools/apksigner" sign --ks "$apk_dir/debug.keystore" --ks-pass pass:android --key-pass pass:android \
  --out "$apk_dir/itu-audio-debug.apk" "$apk_dir/aligned.apk"
"$tools/apksigner" verify --verbose "$apk_dir/itu-audio-debug.apk"
"$tools/zipalign" -c -P 16 4 "$apk_dir/itu-audio-debug.apk"
echo "$apk_dir/itu-audio-debug.apk"
