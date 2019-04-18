
NDKROOT=$1
SYSROOT=$NDKROOT/sysroot
make clean
# Set LDFLAGS so that the linker finds the appropriate libgcc
export LDFLAGS="-L${NDKROOT}/toolchains/x86-4.9/prebuilt/darwin-x86_64/lib/gcc/i686-linux-android/4.9.x -lgcc"

# Set the clang cross compile flags
export CLANG_FLAGS="-target i686-linux-android -D__ANDROID_API__=16 \
  --sysroot=$NDKROOT/platforms/android-16/arch-x86 -isystem $SYSROOT/usr/include/i686-linux-android \
  -isystem $SYSROOT/usr/include -ffunction-sections -fdata-sections \
  -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/libs/x86/include/ -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include/backward \
  --gcc-toolchain=$NDKROOT/toolchains/x86-4.9/prebuilt/darwin-x86_64"
#OpenBLAS Compile
make TARGET=x86 BINARY=32 NOFORTRAN=1 NUM_THREADS=1 AR=$NDKROOT/toolchains/x86-4.9/prebuilt/darwin-x86_64/i686-linux-android/bin/ar CC="$NDKROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang ${CLANG_FLAGS}" HOSTCC=gcc LDFLAGS=$LDFLAGS
make PREFIX=$PWD/output/i686/ NOFORTRAN=0 install
