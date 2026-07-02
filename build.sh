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
	# above. On arm64 macOS, use the committed overlay triplet (cmake/vcpkg-triplets)
	# whose chainload toolchain forces the same compiler + arch — it honors the
	# exported CC/CXX — so boost/protobuf build with the fallback compiler too.
	# Same overlay is used by CMakePresets.json (VS Code / fresh clone).
	if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
		CMAKE_ARGS+=(-DVCPKG_OVERLAY_TRIPLETS="$SCRIPT_DIR/cmake/vcpkg-triplets" -DVCPKG_TARGET_TRIPLET=arm64-osx)
		PRESET_OVERLAY=1
	fi
fi

# Emit a machine-local CMakeUserPresets.json (gitignored) mirroring the exact
# configuration above (absolute vcpkg/compiler paths for this machine). Committed
# CMakePresets.json already provides a portable "macos-arm64" preset; this just
# pins the probed compiler + resolved vcpkg toolchain for local IDE convenience.
CACHE_VARS="\"CMAKE_BUILD_TYPE\": \"${BUILD_TYPE:-Release}\""
if [ -n "${CXX:-}" ]; then
	CACHE_VARS="$CACHE_VARS,
				\"CMAKE_C_COMPILER\": \"${CC:-cc}\",
				\"CMAKE_CXX_COMPILER\": \"$CXX\""
fi
if [ "${PRESET_OVERLAY:-0}" = "1" ]; then
	CACHE_VARS="$CACHE_VARS,
				\"VCPKG_TARGET_TRIPLET\": \"arm64-osx\",
				\"VCPKG_OVERLAY_TRIPLETS\": \"\${sourceDir}/cmake/vcpkg-triplets\""
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
