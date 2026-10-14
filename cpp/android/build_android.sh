#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "$0")/../.." && pwd)
build_dir=${1:-"$project_root/build/android-arm64"}
sdk_root=${ANDROID_SDK_ROOT:-"$HOME/.android-sdk"}
ndk_root=${ANDROID_NDK_HOME:-"$sdk_root/ndk/27.2.12479018"}
if [ "$(uname -m)" != x86_64 ]; then
  echo 'Official Linux NDK needs an x86_64 build host. Run this script on myserver.' >&2
  exit 1
fi
test -f "$ndk_root/build/cmake/android.toolchain.cmake"
cmake -S "$project_root" -B "$build_dir" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ndk_root/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DANDROID_STL=c++_static -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
  -DITU_RNNOISE_SOURCE_DIR="${ITU_RNNOISE_SOURCE_DIR:-}" \
  -DITU_ORT_ROOT="${ITU_ORT_ROOT:-}" \
  -DITU_SPEEXDSP_SOURCE_DIR="${ITU_SPEEXDSP_SOURCE_DIR:-$project_root/build/deps/speexdsp-1.2.1}" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build "$build_dir" --parallel 4
"$ndk_root/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf" -l "$build_dir/cpp/libitu_audio_native.so"
"$ndk_root/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf" -d "$build_dir/cpp/libitu_audio_native.so"
