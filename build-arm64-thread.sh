
NDKROOT=$1
SYSROOT=$NDKROOT/sysroot
make clean
# Set LDFLAGS so that the linker finds the appropriate libgcc
export LDFLAGS="-L${NDKROOT}/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64/lib/gcc/aarch64-linux-android/4.9.x -lm"

# Set the clang cross compile flags
export CLANG_FLAGS="-target aarch64-linux-android -D__ANDROID_API__=21 \
  --sysroot=$NDKROOT/platforms/android-23/arch-arm64 -isystem $SYSROOT/usr/include/aarch64-linux-android \
  -isystem $SYSROOT/usr/include -ffunction-sections -fdata-sections \
  -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/include/ -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include/backward \
  --gcc-toolchain=$NDKROOT/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64"
#OpenBLAS Compile
make TARGET=CORTEXA57 NOFORTRAN=1 USE_THREAD=1 NUM_THREADS=4 USE_LOCKING=1 AR=$NDKROOT/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64/aarch64-linux-android/bin/ar CC="$NDKROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang ${CLANG_FLAGS}" HOSTCC=gcc
make PREFIX=$PWD/output/arm64_multi_thread/ NOFORTRAN=0 install
