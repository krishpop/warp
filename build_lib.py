# This script is an 'offline' build of the core warp runtime libraries
# designed to be executed as part of CI / developer workflows, not 
# as part of the user runtime (since it requires CUDA toolkit, etc)

import sys
if sys.version_info[0] < 3:
    raise Exception("Warp requires Python 3.x minimum")

import os
import argparse

import warp.config
import warp.build

parser = argparse.ArgumentParser(description="Warp build script")
parser.add_argument('--msvc_path', type=str, help='Path to MSVC compiler (optional if already on PATH)')
parser.add_argument('--sdk_path', type=str, help='Path to WinSDK (optional if already on PATH)')
parser.add_argument('--cuda_path', type=str, help='Path to CUDA SDK')
parser.add_argument('--mode', type=str, default="release", help="Build configuration, either 'release' or 'debug'")
parser.add_argument('--verbose', type=bool, default=True, help="Verbose building output, default True")
parser.add_argument('--verify_fp', type=bool, default=False, help="Verify kernel inputs and outputs are finite after each launch, default False")
parser.add_argument('--fast_math', type=bool, default=False, help="Enable fast math on library, default False")
parser.add_argument('--all_architectures', action='store_true', help="Generate code for all supported architectures")
args = parser.parse_args()

# set build output path off this file
build_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "warp")

print(args)

warp.config.verbose = args.verbose
warp.config.mode = args.mode
warp.config.verify_fp = args.verify_fp
warp.config.fast_math = args.fast_math


# See PyTorch for reference on how to find nvcc.exe more robustly, https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension
def find_cuda():
    
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    return cuda_home


# setup CUDA paths
if sys.platform == 'darwin':

    warp.config.cuda_path = None

else:

    if args.cuda_path:
        warp.config.cuda_path = args.cuda_path
    else:
        warp.config.cuda_path = find_cuda()


# setup MSVC and WinSDK paths
if os.name == 'nt':
    
    if args.sdk_path and args.msvc_path:
        # user provided MSVC
        warp.build.set_msvc_compiler(msvc_path=args.msvc_path, sdk_path=args.sdk_path)
    else:
        
        # attempt to find MSVC in environment (will set vcvars)
        warp.config.host_compiler = warp.build.find_host_compiler()
        
        if not warp.config.host_compiler:
            print("Warp build error: Could not find MSVC compiler")
            sys.exit(1)


try:

    cpp_sources = [
        "native/warp.cpp",
        "native/crt.cpp",
        "native/cuda_util.cpp",
        "native/mesh.cpp",
        "native/hashgrid.cpp",
        "native/sort.cpp",
        "native/volume.cpp",
        "native/marching.cpp",
        "native/cutlass_gemm.cpp",
    ]
    warp_cpp_paths = [os.path.join(build_path, cpp) for cpp in cpp_sources]

    if (warp.config.cuda_path is None):
        print("Warning: CUDA toolchain not found, building without CUDA support")
        warp_cu_path = None
    else:
        warp_cu_path = os.path.join(build_path, "native/warp.cu")

    if sys.platform == "win32":
        dll_path = os.path.join(build_path, "bin/warp.dll")
    elif sys.platform == "darwin":
        dll_path = os.path.join(build_path, "bin/warp.dylib")
    else:
        dll_path = os.path.join(build_path, "bin/warp.so")

    warp.build.build_dll(
                    dll_path=dll_path,
                    cpp_paths=warp_cpp_paths,
                    cu_path=warp_cu_path,
                    mode=warp.config.mode,
                    verify_fp=warp.config.verify_fp,
                    fast_math=args.fast_math,
                    use_cache=False,
                    all_architectures=args.all_architectures)
                    
except Exception as e:

    # output build error
    print(f"Warp build error: {e}")

    # report error
    sys.exit(1)
