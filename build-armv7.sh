
NDKROOT=$1
SYSROOT=$NDKROOT/sysroot
make clean
# Set LDFLAGS so that the linker finds the appropriate libgcc
export LDFLAGS="-L${NDKROOT}/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/lib/gcc/arm-linux-androideabi/4.9.x -lgcc"

# Set the clang cross compile flags
export CLANG_FLAGS="-target armv7-linux-androideabi -D__ANDROID_API__=16 -marm -mfpu=vfp -mfloat-abi=softfp \
  --sysroot=$NDKROOT/platforms/android-16/arch-arm -isystem $SYSROOT/usr/include/arm-linux-androideabi \
  -isystem $SYSROOT/usr/include -ffunction-sections -fdata-sections \
  -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/include/ -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include/backward \
  --gcc-toolchain=$NDKROOT/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64"
#OpenBLAS Compile
make TARGET=ARMV7 NOFORTRAN=1 USE_THREAD=0 NUM_THREADS=1 AR=$NDKROOT/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/arm-linux-androideabi/bin/ar CC="$NDKROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang ${CLANG_FLAGS}" HOSTCC=gcc ARM_SOFTFP_ABI=1 LDFLAGS=$LDFLAGS
make PREFIX=$PWD/output/armv7/ NOFORTRAN=0 install
