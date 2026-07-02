# vcpkg chainload toolchain (committed, shared) — forces a C++ compiler that can
# build the dependency ports. Recent macOS SDK libc++ (<charconv>/<format>) uses
# __builtin_clzg, which Apple clang 16 lacks, so C++23 fails to compile; Homebrew
# LLVM clang provides it. Honors CC/CXX when set (build.sh probe), otherwise uses
# the standard Homebrew LLVM location (brew install llvm).
if(DEFINED ENV{CC})
	set(CMAKE_C_COMPILER "$ENV{CC}" CACHE FILEPATH "" FORCE)
elseif(EXISTS "/opt/homebrew/opt/llvm/bin/clang")
	set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang" CACHE FILEPATH "" FORCE)
endif()

if(DEFINED ENV{CXX})
	set(CMAKE_CXX_COMPILER "$ENV{CXX}" CACHE FILEPATH "" FORCE)
elseif(EXISTS "/opt/homebrew/opt/llvm/bin/clang++")
	set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++" CACHE FILEPATH "" FORCE)
endif()

# A chainload toolchain replaces vcpkg's osx toolchain, so set arch here too or
# boost-context selects x86_64 assembly on an arm64 build and fails to assemble.
set(CMAKE_SYSTEM_PROCESSOR arm64 CACHE STRING "" FORCE)
set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "" FORCE)
