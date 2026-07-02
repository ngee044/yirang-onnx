#!/bin/bash
#
# Build yirang-onnx (Release) with the vcpkg toolchain + Ninja.
#
# Environment overrides (all optional):
#   VCPKG_TOOLCHAIN   path to vcpkg.cmake (default: $HOME/vcpkg/scripts/buildsystems/vcpkg.cmake)
#   BUILD_TYPE        CMake build type    (default: Release)
#   CC / CXX          force a compiler    (default: probed — see below)
#   FRESH_DEPS=1      wipe build/ so vcpkg reinstalls from the manifest (after editing vcpkg.json)
#
# Probes the system compiler; if it cannot build C++23 (Apple clang lacks the
# __builtin_clzg the macOS SDK libc++ needs) it falls back to Homebrew LLVM clang.
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
VCPKG_TOOLCHAIN="${VCPKG_TOOLCHAIN:-$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake}"

if [ ! -f "$VCPKG_TOOLCHAIN" ]; then
	echo "[build.sh] vcpkg toolchain not found: $VCPKG_TOOLCHAIN (override with VCPKG_TOOLCHAIN=...)" >&2
	exit 1
fi

if [ ! -d "$SCRIPT_DIR/.CppToolkit/Utilities" ]; then
	echo "[build.sh] .CppToolkit submodule missing; run: git submodule update --init --recursive" >&2
	exit 1
fi

if [ "${FRESH_DEPS:-0}" = "1" ]; then
	echo "[build.sh] FRESH_DEPS=1 -> removing $BUILD_DIR"
	rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

probe_cxx() {
	printf '#include <format>\nint main(){auto s=std::format("{}",1);return (int)s.size();}\n' \
		| "$1" -std=c++23 -x c++ - -o "$BUILD_DIR/.cxxprobe" >/dev/null 2>&1
	local rc=$?
	rm -f "$BUILD_DIR/.cxxprobe"
	return $rc
}

if [ -z "${CXX:-}" ]; then
	if probe_cxx "c++"; then
		echo "[build.sh] using system compiler (c++)"
	elif [ -x /opt/homebrew/opt/llvm/bin/clang++ ] && probe_cxx /opt/homebrew/opt/llvm/bin/clang++; then
		export CXX=/opt/homebrew/opt/llvm/bin/clang++
		export CC=/opt/homebrew/opt/llvm/bin/clang
		echo "[build.sh] system c++ cannot build -std=c++23 (macOS SDK libc++ / __builtin_clzg); using Homebrew LLVM: $CXX"
	else
		echo "[build.sh] no available compiler can build -std=c++23; install Homebrew LLVM: brew install llvm" >&2
		exit 1
	fi
fi

CMAKE_ARGS=(
	-S "$SCRIPT_DIR" -B "$BUILD_DIR" -G Ninja
	-DCMAKE_TOOLCHAIN_FILE="$VCPKG_TOOLCHAIN"
	-DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}"
)

if [ -n "${CXX:-}" ]; then
	CMAKE_ARGS+=(-DCMAKE_C_COMPILER="${CC:-cc}" -DCMAKE_CXX_COMPILER="$CXX")

	# vcpkg builds dependency ports in separate CMake runs that ignore the flags
	# above. On arm64 macOS we generate a throwaway overlay triplet under build/
	# (gitignored) whose chainload toolchain forces the same compiler + arch, so
	# boost/protobuf build with the fallback compiler too. Nothing is committed.
	if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
		OVERLAY_DIR="$BUILD_DIR/vcpkg-overlay"
		mkdir -p "$OVERLAY_DIR"
		cat > "$OVERLAY_DIR/toolchain.cmake" <<EOF
set(CMAKE_C_COMPILER "${CC:-cc}" CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "$CXX" CACHE FILEPATH "" FORCE)
set(CMAKE_SYSTEM_PROCESSOR arm64 CACHE STRING "" FORCE)
set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "" FORCE)
EOF
		cat > "$OVERLAY_DIR/arm64-osx.cmake" <<EOF
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_OSX_ARCHITECTURES arm64)
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "$OVERLAY_DIR/toolchain.cmake")
EOF
		CMAKE_ARGS+=(-DVCPKG_OVERLAY_TRIPLETS="$OVERLAY_DIR" -DVCPKG_TARGET_TRIPLET=arm64-osx)
		PRESET_OVERLAY=1
	fi
fi

# Emit a machine-local CMakeUserPresets.json (gitignored) mirroring the exact
# configuration above, so IDEs (VS Code / CLion CMake Tools) configure with the
# same compiler + vcpkg overlay instead of the system Apple clang, which cannot
# build the C++23 stdlib (macOS SDK libc++ needs __builtin_clzg/__builtin_ctzg,
# absent in Apple clang). Regenerated every run so paths stay in sync; requires
# this script to have run once so build/vcpkg-overlay exists.
CACHE_VARS="\"CMAKE_BUILD_TYPE\": \"${BUILD_TYPE:-Release}\""
if [ -n "${CXX:-}" ]; then
	CACHE_VARS="$CACHE_VARS,
				\"CMAKE_C_COMPILER\": \"${CC:-cc}\",
				\"CMAKE_CXX_COMPILER\": \"$CXX\""
fi
if [ "${PRESET_OVERLAY:-0}" = "1" ]; then
	CACHE_VARS="$CACHE_VARS,
				\"VCPKG_TARGET_TRIPLET\": \"arm64-osx\",
				\"VCPKG_OVERLAY_TRIPLETS\": \"\${sourceDir}/build/vcpkg-overlay\""
fi
cat > "$SCRIPT_DIR/CMakeUserPresets.json" <<EOF
{
	"version": 3,
	"configurePresets": [
		{
			"name": "yirang-onnx",
			"displayName": "yirang-onnx (build.sh)",
			"generator": "Ninja",
			"binaryDir": "\${sourceDir}/build",
			"toolchainFile": "$VCPKG_TOOLCHAIN",
			"cacheVariables": {
				$CACHE_VARS
			}
		}
	],
	"buildPresets": [
		{ "name": "yirang-onnx", "configurePreset": "yirang-onnx" }
	]
}
EOF

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j
