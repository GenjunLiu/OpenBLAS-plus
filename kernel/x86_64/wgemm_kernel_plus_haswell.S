/*********************************************************************************
Copyright (c) 2013, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************/

/*********************************************************************************
ARGS:
                   C             LINUX          WINDOWS_ABI
        ARG1    M(LONG)        ARG1(%rdi)        ARG1(%rcx)
        ARG2    N(LONG)        ARG2(%rsi)        ARG2(%rdx)
        ARG3    K(LONG)        ARG3(%rdx)        ARG3(%r8)
        ARG4    ALPHA(INT16)   ARG4(%rcx)        ARG4(%r9)
        ARG5    A(INT16*)      ARG5(%r8)          40(%rsp)
        ARG6    LDA(LONG)      ARG6(%r9)          48(%rsp)
        ARG7    B(INT16*)        8(%rsp)          56(%rsp)
        ARG8    LDB(LONG)       16(%rsp)          64(%rsp)
        ARG9    C(INT16*)       24(%rsp)          72(%rsp)
        ARG10   LDC(LONG)       32(%rsp)          80(%rsp)


LINUX:       stack offset 8  = return adderss
WINDOWS_ABI: stack offset 40 = return adderss + %rcx + %rdx + %r8 + %r9
**********************************************************************************/

#define ASSEMBLER
#include "common.h"

#ifndef WINDOWS_ABI /* Linux, Darwin, Unix */

#define STACKSIZE 96

#define M               %rdi
#define N               %rsi
#define K               %rdx
#define ALPHA           %rcx
#define A               %r8
#define LDA             %r9
#define OLD_B            8 + STACKSIZE(%rsp)
#define OLD_LDB         16 + STACKSIZE(%rsp)
#define OLD_C           24 + STACKSIZE(%rsp)
#define OLD_LDC         32 + STACKSIZE(%rsp)

#define B               %r10
#define LDB             %r11
#define C               %r12
#define LDC             %r13
#define ALPHA_B         %cl

#define SP              %r14
#define AO              %r8   /* A */
#define BO              %r15
#define I               %rbx
#define J               %rbp
#define KK              %rax
#define KKK             %r9   /* same with LDA, LDA will be store in stack */
#define RESULT          %r9d  /* same with KKK */
#define RESULT_W        %r9w
#define LDA_STACK       8(%rsp)

#else  /* Windows x64 */

#define STACKSIZE 256

#define OLD_M           %rcx
#define N               %rdx
#define K               %r8
#define OLD_ALPHA       %r9
#define OLD_A           40 + STACKSIZE(%rsp)
#define OLD_LDA         48 + STACKSIZE(%rsp)
#define OLD_B           56 + STACKSIZE(%rsp)
#define OLD_LDB         64 + STACKSIZE(%rsp)
#define OLD_C           72 + STACKSIZE(%rsp)
#define OLD_LDC         80 + STACKSIZE(%rsp)

#define M               %r9
#define ALPHA           %rcx
#define A               %r10
#define LDA             %r11
#define B               %r12
#define LDB             %r13
#define C               %r14
#define LDC             %r15
#define ALPHA_B         %cl

#define SP              %rbx
#define AO              %r10  /* A */
#define BO              %rdi
#define I               %rbp
#define J               %rbi
#define KK              %rax
#define KKK             %r11  /* same with LDA, LDA will be store in stack */
#define RESULT          %r11d /* same with KKK */
#define RESULT_W        %r11w
#define LDA_STACK       8(%rsp)

#endif

#define A_PR1           512
#define B_PR1           512

#if defined(OS_WINDOWS)
#define L_BUFFER_SIZE 8192
#else
#define L_BUFFER_SIZE 12288
#endif

#if defined(OS_WINDOWS)
#if   L_BUFFER_SIZE > 16384
#define STACK_TOUCH \
        movl    $0,  4096 * 4(%rsp);\
        movl    $0,  4096 * 3(%rsp);\
        movl    $0,  4096 * 2(%rsp);\
        movl    $0,  4096 * 1(%rsp);
#elif L_BUFFER_SIZE > 12288
#define STACK_TOUCH \
        movl    $0,  4096 * 3(%rsp);\
        movl    $0,  4096 * 2(%rsp);\
        movl    $0,  4096 * 1(%rsp);
#elif L_BUFFER_SIZE > 8192
#define STACK_TOUCH \
        movl    $0,  4096 * 2(%rsp);\
        movl    $0,  4096 * 1(%rsp);
#elif L_BUFFER_SIZE > 4096
#define STACK_TOUCH \
        movl    $0,  4096 * 1(%rsp);
#else
#define STACK_TOUCH
#endif
#else
#define STACK_TOUCH
#endif

/*******************************************************************************************/

.macro KERNEL_MUL_6x6_SUB

    vmovdqu      0 * SIZE(AO), %ymm2
    vmovdqu     16 * SIZE(AO), %ymm3
    vmovdqu     32 * SIZE(AO), %ymm4
    vmovdqu     48 * SIZE(AO), %ymm5
    vmovdqu     64 * SIZE(AO), %ymm6
    vmovdqu     80 * SIZE(AO), %ymm7

    vmovdqu      0 * SIZE(BO), %ymm10
    vmovdqu     16 * SIZE(BO), %ymm11
    vmovdqu     32 * SIZE(BO), %ymm12
    vmovdqu     48 * SIZE(BO), %ymm13
    vmovdqu     64 * SIZE(BO), %ymm14
    vmovdqu     80 * SIZE(BO), %ymm15

    vpmaddwd   %ymm2, %ymm10, %ymm2
    vpmaddwd   %ymm3, %ymm11, %ymm3
    vpmaddwd   %ymm4, %ymm12, %ymm4
    vpmaddwd   %ymm5, %ymm13, %ymm5
    vpmaddwd   %ymm6, %ymm14, %ymm6
    vpmaddwd   %ymm7, %ymm15, %ymm7

    # format must be  `$ 96*SIZE, AO`, should not be `$96*SZIE, AO` or `$ 96 * SIZE, AO`
    addq  $ 96*SIZE, AO
    addq  $ 96*SIZE, BO

.endm

/*******************************************************************************************/

.macro KERNEL_MUL_4x4_SUB

    vmovdqu      0 * SIZE(AO), %ymm4
    vmovdqu     16 * SIZE(AO), %ymm5
    vmovdqu     32 * SIZE(AO), %ymm6
    vmovdqu     48 * SIZE(AO), %ymm7

    vmovdqu      0 * SIZE(BO), %ymm12
    vmovdqu     16 * SIZE(BO), %ymm13
    vmovdqu     32 * SIZE(BO), %ymm14
    vmovdqu     48 * SIZE(BO), %ymm15

    vpmaddwd   %ymm4, %ymm12, %ymm4
    vpmaddwd   %ymm5, %ymm13, %ymm5
    vpmaddwd   %ymm6, %ymm14, %ymm6
    vpmaddwd   %ymm7, %ymm15, %ymm7

    addq  $ 64*SIZE, AO
    addq  $ 64*SIZE, BO

.endm

/*******************************************************************************************/

.macro KERNEL_MUL_2x2_SUB

    vmovdqu      0 * SIZE(AO), %ymm6
    vmovdqu     16 * SIZE(AO), %ymm7

    vmovdqu      0 * SIZE(BO), %ymm14
    vmovdqu     16 * SIZE(BO), %ymm15

    vpmaddwd   %ymm6, %ymm14,  %ymm6
    vpmaddwd   %ymm7, %ymm15,  %ymm7

    addq  $ 32*SIZE, AO
    addq  $ 32*SIZE, BO

.endm

/*******************************************************************************************/

.macro KERNEL_MUL_1x1_SUB

    vmovdqu      0 * SIZE(AO), %ymm7
    vmovdqu      0 * SIZE(BO), %ymm15

    vpmaddwd   %ymm7, %ymm15,  %ymm7

    addq  $ 16*SIZE, AO
    addq  $ 16*SIZE, BO

.endm

/*******************************************************************************************/

.macro KERNEL256_SUB

    KERNEL_MUL_6x6_SUB

    vpaddd %ymm2,  %ymm3,  %ymm1
    vpaddd %ymm4,  %ymm5,  %ymm8
    vpaddd %ymm6,  %ymm7,  %ymm9         # %ymm1, %ymm8, %ymm9 local result, %ymm0 global result

    KERNEL_MUL_6x6_SUB

    vpaddd %ymm2,  %ymm3,  %ymm2
    vpaddd %ymm4,  %ymm5,  %ymm3
    vpaddd %ymm6,  %ymm7,  %ymm10        # %ymm1~3, %ymm8~10 local result, %ymm0 global result

    KERNEL_MUL_4x4_SUB

    vpaddd %ymm1,  %ymm6,   %ymm1
    vpaddd %ymm2,  %ymm7,   %ymm2
    vpaddd %ymm3,  %ymm8,   %ymm3
    vpaddd %ymm4,  %ymm9,   %ymm4
    vpaddd %ymm5,  %ymm10,  %ymm5

    vpaddd %ymm1,  %ymm2,   %ymm1
    vpaddd %ymm3,  %ymm4,   %ymm3

    vpaddd %ymm1,  %ymm3,   %ymm1
    vpaddd %ymm1,  %ymm5,   %ymm1

    vpaddd %ymm1,  %ymm0,  %ymm0

.endm

/*******************************************************************************************/

.macro KERNEL128_SUB

    KERNEL_MUL_4x4_SUB

    vpaddd %ymm4,  %ymm5,  %ymm2
    vpaddd %ymm6,  %ymm7,  %ymm3

    KERNEL_MUL_4x4_SUB

    vpaddd %ymm4,  %ymm6,  %ymm10
    vpaddd %ymm5,  %ymm7,  %ymm11

    vpaddd %ymm2,  %ymm10, %ymm2   # 4 x %ymm --> 2 x %ymm
    vpaddd %ymm3,  %ymm11, %ymm3

    vpaddd %ymm2,  %ymm3,  %ymm2   # 2 x %ymm --> 1 x %ymm

    vpaddd %ymm2,  %ymm0,  %ymm0

.endm

/*******************************************************************************************/

.macro KERNEL64_SUB

    KERNEL_MUL_4x4_SUB

    vpaddd %ymm4,  %ymm5,  %ymm2
    vpaddd %ymm6,  %ymm7,  %ymm3

    vpaddd %ymm2,  %ymm3,  %ymm2

    vpaddd %ymm2,  %ymm0,  %ymm0

.endm

/*******************************************************************************************/

.macro KERNEL32_SUB

    KERNEL_MUL_2x2_SUB

    vpaddd %ymm6,  %ymm7,  %ymm6

    vpaddd %ymm6,  %ymm0,  %ymm0

.endm

/*******************************************************************************************/

.macro KERNEL16_SUB

    KERNEL_MUL_1x1_SUB

    vpaddd %ymm7,  %ymm0,  %ymm0

.endm

/*******************************************************************************************/

.macro KERNEL8_SUB

    movdqu      0 * SIZE(AO), %xmm6
    movdqu      0 * SIZE(BO), %xmm7

    pmaddwd   %xmm7, %xmm6

    paddd     %xmm6, %xmm0

    addq  $ 8*SIZE, AO
    addq  $ 8*SIZE, BO

.endm

/*******************************************************************************************/

.macro KERNEL4_SUB
    
    # xmm[0] = (AO) xmm[1] = 4(AO) xmm[2] = zero xmm[3] = zero
    movq      0 * SIZE(AO), %xmm6
    movq      0 * SIZE(BO), %xmm7

    pmaddwd   %xmm7, %xmm6

    paddd     %xmm6, %xmm0

    addq  $ 4*SIZE, AO
    addq  $ 4*SIZE, BO

.endm

/*******************************************************************************************/

.macro KERNEL2_SUB

    # xmm[0] = (AO) xmm[1] = zero xmm[2] = zero xmm[3] = zero
    movd      0 * SIZE(AO), %xmm6
    movd      0 * SIZE(BO), %xmm7

    pmaddwd   %xmm7, %xmm6

    paddd     %xmm6, %xmm0

    addq  $ 2*SIZE, AO
    addq  $ 2*SIZE, BO

.endm

/*******************************************************************************************/

.macro KERNEL1_SUB

    push %rax
    push %rbx
    movswl    0 * SIZE(AO), %eax
    movswl    0 * SIZE(BO), %ebx
    imull %eax, %ebx
    movd %ebx, %xmm9
    pop  %rbx
    pop  %rax

    paddd     %xmm9, %xmm0

    addq  $ 1*SIZE, AO
    addq  $ 1*SIZE, BO
.endm

/*******************************************************************************************/

.macro SAVE_KxK_SUB

    # 8*32 result in %ymm15
    vextractf128 $ 1, %ymm0, %xmm1     # %ymm0 hi 128bit --> %xmm1
    paddd     %xmm1, %xmm0             # %ymm0 = %ymm1 + %ymm0
    phaddd    %xmm0, %xmm0             # xmm0[0] = xmm0[0] + xmm0[1], xmm0[1] = xmm0[2] + xmm0[3]
    phaddd    %xmm0, %xmm0
    pextrd    $ 0, %xmm0, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(C)
    addq      $ 1*SIZE, C

.endm

/*******************************************************************************************
* WGEMM Plus Kernel
*******************************************************************************************/

    PROLOGUE
    PROFCODE

    subq    $STACKSIZE, %rsp
    movq    %rbx,      (%rsp)
    movq    %rbp,     8(%rsp)
    movq    %r12,    16(%rsp)
    movq    %r13,    24(%rsp)
    movq    %r14,    32(%rsp)
    movq    %r15,    40(%rsp)

    vzeroupper

#ifdef WINDOWS_ABI
    movq    %rdi,    48(%rsp)
    movq    %rsi,    56(%rsp)
    movups  %xmm6,   64(%rsp)
    movups  %xmm7,   80(%rsp)
    movups  %xmm8,   96(%rsp)
    movups  %xmm9,  112(%rsp)
    movups  %xmm10, 128(%rsp)
    movups  %xmm11, 144(%rsp)
    movups  %xmm12, 160(%rsp)
    movups  %xmm13, 176(%rsp)
    movups  %xmm14, 192(%rsp)
    movups  %xmm15, 208(%rsp)

    # swap  M, ALPHA
    movq    OLD_ALPHA,  224(%rsp)
    movq    OLD_M, M
    movq    224(%rsp), ALPHA

    movq    OLD_A,  A
    movq    OLD_LDA,LDA
#endif

    movq    OLD_B,  B
    movq    OLD_LDB,LDB
    movq    OLD_C,  C
    movq    OLD_LDC,LDC

    movq    %rsp, SP      # save old stack
    subq    $128 + L_BUFFER_SIZE, %rsp
    andq    $-4096, %rsp  # align stack

    STACK_TOUCH
    
    cmpq    $0, M
    je    .L999

    cmpq    $0, N
    je    .L999

    cmpq    $0, K

    #subq    K, LDA             # LDA = LDA - K
    subq    K, LDB             # LDB = LDB - K
    subq    N, LDC             # LDC = LDC - N
    salq    $BASE_SHIFT, LDA
    salq    $BASE_SHIFT, LDB
    salq    $BASE_SHIFT, LDC
    movq    LDA, LDA_STACK     # save LDA

    movq    M, I

.L0_0:
    #movq    A, AO               # AO = A
    movq    B, BO               # BO = B
    movq    N, J

.L1_0:
    # AO x BO -> C, KERNEL
    vxorps  %ymm0, %ymm0, %ymm0 # %ymm0 = 0 
    movq    K, KK
    sarq    $8, KK              # KK = K>>8
    jz .L2_1

.L2_0:

    KERNEL256_SUB

    decq    KK
    jnz .L2_0

    ALIGN_4

.L2_1:

    movq    K, KKK
    andq    $255, KKK           # KKK = K%256
    jz .L2_9

    testq   $128, KKK
    jz .L2_2

    KERNEL128_SUB

    ALIGN_4

.L2_2:
    testq   $64, KKK
    jz .L2_3

    KERNEL64_SUB

    ALIGN_4

.L2_3:
    testq   $32, KKK
    jz .L2_4

    KERNEL32_SUB

    ALIGN_4

.L2_4:
    testq   $16, KKK
    jz .L2_5

    KERNEL16_SUB

    ALIGN_4

.L2_5:
    testq   $8, KKK
    jz .L2_6

    KERNEL8_SUB

    ALIGN_4

.L2_6:
    testq   $4, KKK
    jz .L2_7

    KERNEL4_SUB

    ALIGN_4

.L2_7:
    testq   $2, KKK
    jz .L2_8

    KERNEL2_SUB

    ALIGN_4

.L2_8:
    testq   $1, KKK
    jz .L2_9

    KERNEL1_SUB

    ALIGN_4

.L2_9:
    SAVE_KxK_SUB

.L1_1:
    neg     K
    leaq    (AO, K, SIZE), AO   # AO = AO - K
    neg     K
    leaq    (BO, LDB, 1), BO    # B = B + LDB
    decq    J
    jnz     .L1_0               # j != 0

.L0_1:
    prefetcht0 LDA_STACK
    movq    LDA_STACK, LDA      # get LDA from stack
    leaq    (AO, LDA, 1), AO    # AO = AO + LDA
    leaq    (C,  LDC, 1), C     # C = C + LDA
    decq    I
    jnz     .L0_0

.L999:
    movq           SP, %rsp
    movq       (%rsp), %rbx
    movq      8(%rsp), %rbp
    movq     16(%rsp), %r12
    movq     24(%rsp), %r13
    movq     32(%rsp), %r14
    movq     40(%rsp), %r15

#ifdef WINDOWS_ABI
    movq     48(%rsp), %rdi
    movq     56(%rsp), %rsi
    movups   64(%rsp), %xmm6
    movups   80(%rsp), %xmm7
    movups   96(%rsp), %xmm8
    movups  112(%rsp), %xmm9
    movups  128(%rsp), %xmm10
    movups  144(%rsp), %xmm11
    movups  160(%rsp), %xmm12
    movups  176(%rsp), %xmm13
    movups  192(%rsp), %xmm14
    movups  208(%rsp), %xmm15
#endif

    addq    $STACKSIZE, %rsp
    ret

    EPILOGUE
