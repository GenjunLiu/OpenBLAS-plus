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

#define J               %rax
#define AO1             %rbx
#define AO2             %rbp
#define CO1             %r15

#define CO2             SP
#define BO1             A
#define BO2             LDA
#define BO3             C
#define BO4             LDC
#define KK              B

#define TEMP            %rdi
#define TEMP_D          %edi
#define TEMP_W          %di

#define RESULT          KK
#define RESULT_W        %r10w

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

#define J               %rax
#define AO1             %rbp
#define AO2             %rsi
#define CO1             %rdi

#define CO2             SP
#define BO1             A
#define BO2             LDA
#define BO3             C
#define BO4             LDC
#define KK              B

#define TEMP            %r9
#define TEMP_D          %r9d
#define TEMP_W          %r9w

#define RESULT          KK
#define RESULT_W        %r12w

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

.macro KERNEL2x4_32_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8
    vmovdqu     0 * SIZE(AO2), %ymm10

    vmovdqu     0 * SIZE(BO1), %ymm12
    vmovdqu     0 * SIZE(BO2), %ymm13
    vmovdqu     0 * SIZE(BO3), %ymm14
    vmovdqu     0 * SIZE(BO4), %ymm15

    vpmaddwd    %ymm8,  %ymm12, %ymm9
    vpmaddwd    %ymm8,  %ymm13, %ymm11
    vpaddd      %ymm9,  %ymm0,  %ymm0
    vpaddd      %ymm11, %ymm1,  %ymm1

    vpmaddwd    %ymm8,  %ymm14, %ymm9
    vpmaddwd    %ymm8,  %ymm15, %ymm11
    vpaddd      %ymm9,  %ymm2, %ymm2
    vpaddd      %ymm11, %ymm3, %ymm3

    vpmaddwd    %ymm10, %ymm12, %ymm12
    vpmaddwd    %ymm10, %ymm13, %ymm13
    vpmaddwd    %ymm10, %ymm14, %ymm14
    vpmaddwd    %ymm10, %ymm15, %ymm15

    vpaddd      %ymm12, %ymm4, %ymm4
    vpaddd      %ymm13, %ymm5, %ymm5
    vpaddd      %ymm14, %ymm6, %ymm6
    vpaddd      %ymm15, %ymm7, %ymm7

    vmovdqu    16 * SIZE(AO1), %ymm8
    vmovdqu    16 * SIZE(AO2), %ymm10

    vmovdqu    16 * SIZE(BO1), %ymm12
    vmovdqu    16 * SIZE(BO2), %ymm13
    vmovdqu    16 * SIZE(BO3), %ymm14
    vmovdqu    16 * SIZE(BO4), %ymm15

    vpmaddwd    %ymm8,  %ymm12, %ymm9
    vpmaddwd    %ymm8,  %ymm13, %ymm11
    vpaddd      %ymm9,  %ymm0,  %ymm0
    vpaddd      %ymm11, %ymm1,  %ymm1

    vpmaddwd    %ymm8,  %ymm14, %ymm9
    vpmaddwd    %ymm8,  %ymm15, %ymm11
    vpaddd      %ymm9,  %ymm2, %ymm2
    vpaddd      %ymm11, %ymm3, %ymm3

    vpmaddwd    %ymm10, %ymm12, %ymm12
    vpmaddwd    %ymm10, %ymm13, %ymm13
    vpmaddwd    %ymm10, %ymm14, %ymm14
    vpmaddwd    %ymm10, %ymm15, %ymm15

    vpaddd      %ymm12, %ymm4, %ymm4
    vpaddd      %ymm13, %ymm5, %ymm5
    vpaddd      %ymm14, %ymm6, %ymm6
    vpaddd      %ymm15, %ymm7, %ymm7

    addq    $ 32*SIZE, AO1
    addq    $ 32*SIZE, AO2
    addq    $ 32*SIZE, BO1
    addq    $ 32*SIZE, BO2
    addq    $ 32*SIZE, BO3
    addq    $ 32*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL2x4_16_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8
    vmovdqu     0 * SIZE(AO2), %ymm10

    vmovdqu     0 * SIZE(BO1), %ymm12
    vmovdqu     0 * SIZE(BO2), %ymm13
    vmovdqu     0 * SIZE(BO3), %ymm14
    vmovdqu     0 * SIZE(BO4), %ymm15

    vpmaddwd    %ymm8,  %ymm12, %ymm9
    vpmaddwd    %ymm8,  %ymm13, %ymm11
    vpaddd      %ymm9,  %ymm0,  %ymm0
    vpaddd      %ymm11, %ymm1,  %ymm1

    vpmaddwd    %ymm8,  %ymm14, %ymm9
    vpmaddwd    %ymm8,  %ymm15, %ymm11
    vpaddd      %ymm9,  %ymm2,  %ymm2
    vpaddd      %ymm11, %ymm3,  %ymm3

    vpmaddwd    %ymm10, %ymm12, %ymm12
    vpmaddwd    %ymm10, %ymm13, %ymm13
    vpmaddwd    %ymm10, %ymm14, %ymm14
    vpmaddwd    %ymm10, %ymm15, %ymm15

    vpaddd      %ymm12, %ymm4,  %ymm4
    vpaddd      %ymm13, %ymm5,  %ymm5
    vpaddd      %ymm14, %ymm6,  %ymm6
    vpaddd      %ymm15, %ymm7,  %ymm7

    addq    $ 16*SIZE, AO1
    addq    $ 16*SIZE, AO2
    addq    $ 16*SIZE, BO1
    addq    $ 16*SIZE, BO2
    addq    $ 16*SIZE, BO3
    addq    $ 16*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL2x4_8_SUB

    movdqu     0 * SIZE(AO1), %xmm8
    movdqu     0 * SIZE(AO2), %xmm10

    movdqu     0 * SIZE(BO1), %xmm12
    movdqu     0 * SIZE(BO2), %xmm13
    movdqu     0 * SIZE(BO3), %xmm14
    movdqu     0 * SIZE(BO4), %xmm15

    movdqu     %xmm12,  %xmm9
    movdqu     %xmm13,  %xmm11

    pmaddwd    %xmm8,   %xmm9
    pmaddwd    %xmm8,   %xmm11
    paddd      %xmm9,   %xmm0
    paddd      %xmm11,  %xmm1

    movdqu     %xmm14,  %xmm9
    movdqu     %xmm15,  %xmm11

    pmaddwd    %xmm8,   %xmm9
    pmaddwd    %xmm8,   %xmm11
    paddd      %xmm9,   %xmm2
    paddd      %xmm11,  %xmm3

    pmaddwd    %xmm10,  %xmm12
    pmaddwd    %xmm10,  %xmm13
    pmaddwd    %xmm10,  %xmm14
    pmaddwd    %xmm10,  %xmm15

    paddd      %xmm12,  %xmm4
    paddd      %xmm13,  %xmm5
    paddd      %xmm14,  %xmm6
    paddd      %xmm15,  %xmm7

    addq    $ 8*SIZE, AO1
    addq    $ 8*SIZE, AO2
    addq    $ 8*SIZE, BO1
    addq    $ 8*SIZE, BO2
    addq    $ 8*SIZE, BO3
    addq    $ 8*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL2x4_4_SUB

    movq     0 * SIZE(AO1), %xmm8
    movq     0 * SIZE(AO2), %xmm10

    movq     0 * SIZE(BO1), %xmm12
    movq     0 * SIZE(BO2), %xmm13
    movq     0 * SIZE(BO3), %xmm14
    movq     0 * SIZE(BO4), %xmm15

    movdqu     %xmm12,  %xmm9
    movdqu     %xmm13,  %xmm11

    pmaddwd    %xmm8,   %xmm9
    pmaddwd    %xmm8,   %xmm11
    paddd      %xmm9,   %xmm0
    paddd      %xmm11,  %xmm1

    movdqu     %xmm14,  %xmm9
    movdqu     %xmm15,  %xmm11

    pmaddwd    %xmm8,   %xmm9
    pmaddwd    %xmm8,   %xmm11
    paddd      %xmm9,   %xmm2
    paddd      %xmm11,  %xmm3

    pmaddwd    %xmm10,  %xmm12
    pmaddwd    %xmm10,  %xmm13
    pmaddwd    %xmm10,  %xmm14
    pmaddwd    %xmm10,  %xmm15

    paddd      %xmm12,  %xmm4
    paddd      %xmm13,  %xmm5
    paddd      %xmm14,  %xmm6
    paddd      %xmm15,  %xmm7

    addq    $ 4*SIZE, AO1
    addq    $ 4*SIZE, AO2
    addq    $ 4*SIZE, BO1
    addq    $ 4*SIZE, BO2
    addq    $ 4*SIZE, BO3
    addq    $ 4*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL2x4_2_SUB

    movd     0 * SIZE(AO1), %xmm8
    movd     0 * SIZE(AO2), %xmm10

    movd     0 * SIZE(BO1), %xmm12
    movd     0 * SIZE(BO2), %xmm13
    movd     0 * SIZE(BO3), %xmm14
    movd     0 * SIZE(BO4), %xmm15

    movdqu     %xmm12,  %xmm9
    movdqu     %xmm13,  %xmm11

    pmaddwd    %xmm8,   %xmm9
    pmaddwd    %xmm8,   %xmm11
    paddd      %xmm9,   %xmm0
    paddd      %xmm11,  %xmm1

    movdqu     %xmm14,  %xmm9
    movdqu     %xmm15,  %xmm11

    pmaddwd    %xmm8,   %xmm9
    pmaddwd    %xmm8,   %xmm11
    paddd      %xmm9,   %xmm2
    paddd      %xmm11,  %xmm3

    pmaddwd    %xmm10,  %xmm12
    pmaddwd    %xmm10,  %xmm13
    pmaddwd    %xmm10,  %xmm14
    pmaddwd    %xmm10,  %xmm15

    paddd      %xmm12,  %xmm4
    paddd      %xmm13,  %xmm5
    paddd      %xmm14,  %xmm6
    paddd      %xmm15,  %xmm7

    addq    $ 2*SIZE, AO1
    addq    $ 2*SIZE, AO2
    addq    $ 2*SIZE, BO1
    addq    $ 2*SIZE, BO2
    addq    $ 2*SIZE, BO3
    addq    $ 2*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL2x4_1_SUB

    movzwl    0 * SIZE(AO1), TEMP_D
    movd      TEMP_D, %xmm8
    movzwl    0 * SIZE(AO2), TEMP_D
    movd      TEMP_D, %xmm10

    movzwl    0 * SIZE(BO1), TEMP_D
    movd      TEMP_D, %xmm12
    movzwl    0 * SIZE(BO2), TEMP_D
    movd      TEMP_D, %xmm13
    movzwl    0 * SIZE(BO3), TEMP_D
    movd      TEMP_D, %xmm14
    movzwl    0 * SIZE(BO4), TEMP_D
    movd      TEMP_D, %xmm15

    movdqu     %xmm12,  %xmm9
    movdqu     %xmm13,  %xmm11

    pmaddwd    %xmm8,   %xmm9
    pmaddwd    %xmm8,   %xmm11
    paddd      %xmm9,   %xmm0
    paddd      %xmm11,  %xmm1

    movdqu     %xmm14,  %xmm9
    movdqu     %xmm15,  %xmm11

    pmaddwd    %xmm8,   %xmm9
    pmaddwd    %xmm8,   %xmm11
    paddd      %xmm9,   %xmm2
    paddd      %xmm11,  %xmm3

    pmaddwd    %xmm10,  %xmm12
    pmaddwd    %xmm10,  %xmm13
    pmaddwd    %xmm10,  %xmm14
    pmaddwd    %xmm10,  %xmm15

    paddd      %xmm12,  %xmm4
    paddd      %xmm13,  %xmm5
    paddd      %xmm14,  %xmm6
    paddd      %xmm15,  %xmm7

    addq    $ 1*SIZE, AO1
    addq    $ 1*SIZE, AO2
    addq    $ 1*SIZE, BO1
    addq    $ 1*SIZE, BO2
    addq    $ 1*SIZE, BO3
    addq    $ 1*SIZE, BO4

.endm

/*******************************************************************************************/

.macro RESET2x4_SUB

    vxorps  %ymm0, %ymm0, %ymm0 # %ymm0 = 0
    vxorps  %ymm1, %ymm1, %ymm1 # %ymm1 = 0
    vxorps  %ymm2, %ymm2, %ymm2 # %ymm2 = 0
    vxorps  %ymm3, %ymm3, %ymm3 # %ymm3 = 0
    vxorps  %ymm4, %ymm4, %ymm4 # %ymm4 = 0
    vxorps  %ymm5, %ymm5, %ymm5 # %ymm5 = 0
    vxorps  %ymm6, %ymm6, %ymm6 # %ymm6 = 0
    vxorps  %ymm7, %ymm7, %ymm7 # %ymm7 = 0

.endm

/*******************************************************************************************/

.macro SAVE2x4_SUB

    vextractf128 $ 1, %ymm0, %xmm8     # %ymm0 hi 128bit --> %xmm1
    paddd     %xmm8, %xmm0             # %ymm0 = %ymm1 + %ymm0
    phaddd    %xmm0, %xmm0             # xmm0[0] = xmm0[0] + xmm0[1], xmm0[1] = xmm0[2] + xmm0[3]
    phaddd    %xmm0, %xmm0
    movq      %xmm0, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(CO1)

    vextractf128 $ 1, %ymm1, %xmm8
    paddd     %xmm8, %xmm1
    phaddd    %xmm1, %xmm1
    phaddd    %xmm1, %xmm1
    movq      %xmm1, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 1 * SIZE(CO1)

    vextractf128 $ 1, %ymm2, %xmm8
    paddd     %xmm8, %xmm2
    phaddd    %xmm2, %xmm2
    phaddd    %xmm2, %xmm2
    movq      %xmm2, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 2 * SIZE(CO1)

    vextractf128 $ 1, %ymm3, %xmm8
    paddd     %xmm8, %xmm3
    phaddd    %xmm3, %xmm3
    phaddd    %xmm3, %xmm3
    movq      %xmm3, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 3 * SIZE(CO1)

    vextractf128 $ 1, %ymm4, %xmm8
    paddd     %xmm8, %xmm4
    phaddd    %xmm4, %xmm4
    phaddd    %xmm4, %xmm4
    movq      %xmm4, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(CO2)

    vextractf128 $ 1, %ymm5, %xmm8
    paddd     %xmm8, %xmm5
    phaddd    %xmm5, %xmm5
    phaddd    %xmm5, %xmm5
    movq      %xmm5, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 1 * SIZE(CO2)

    vextractf128 $ 1, %ymm6, %xmm8
    paddd     %xmm8, %xmm6
    phaddd    %xmm6, %xmm6
    phaddd    %xmm6, %xmm6
    movq      %xmm6, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 2 * SIZE(CO2)

    vextractf128 $ 1, %ymm7, %xmm8
    paddd     %xmm8, %xmm7
    phaddd    %xmm7, %xmm7
    phaddd    %xmm7, %xmm7
    movq      %xmm7, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 3 * SIZE(CO2)

    addq      $ 4*SIZE, CO1
    addq      $ 4*SIZE, CO2

.endm

/*******************************************************************************************/

.macro KERNEL2x2_32_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8
    vmovdqu    16 * SIZE(AO1), %ymm9
    vmovdqu     0 * SIZE(AO2), %ymm10
    vmovdqu    16 * SIZE(AO2), %ymm11

    vmovdqu     0 * SIZE(BO1), %ymm12
    vmovdqu    16 * SIZE(BO1), %ymm13
    vmovdqu     0 * SIZE(BO2), %ymm14
    vmovdqu    16 * SIZE(BO2), %ymm15

    vpmaddwd    %ymm8,  %ymm12, %ymm4
    vpmaddwd    %ymm9,  %ymm13, %ymm5
    vpaddd      %ymm4,  %ymm5,  %ymm4
    vpaddd      %ymm4,  %ymm0,  %ymm0

    vpmaddwd    %ymm8,  %ymm14, %ymm4
    vpmaddwd    %ymm9,  %ymm15, %ymm5
    vpaddd      %ymm4,  %ymm5,  %ymm4
    vpaddd      %ymm4,  %ymm1,  %ymm1

    vpmaddwd    %ymm10, %ymm12, %ymm4
    vpmaddwd    %ymm11, %ymm13, %ymm5
    vpaddd      %ymm4,  %ymm5,  %ymm4
    vpaddd      %ymm4,  %ymm2,  %ymm2

    vpmaddwd    %ymm10, %ymm14, %ymm4
    vpmaddwd    %ymm11, %ymm15, %ymm5
    vpaddd      %ymm4,  %ymm5,  %ymm4
    vpaddd      %ymm4,  %ymm3,  %ymm3

    addq    $ 32*SIZE, AO1
    addq    $ 32*SIZE, AO2
    addq    $ 32*SIZE, BO1
    addq    $ 32*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL2x2_16_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8
    vmovdqu     0 * SIZE(AO2), %ymm10

    vmovdqu     0 * SIZE(BO1), %ymm12
    vmovdqu     0 * SIZE(BO2), %ymm14

    vpmaddwd    %ymm8,  %ymm12, %ymm4
    vpaddd      %ymm4,  %ymm0,  %ymm0

    vpmaddwd    %ymm8,  %ymm14, %ymm4
    vpaddd      %ymm4,  %ymm1,  %ymm1

    vpmaddwd    %ymm10, %ymm12, %ymm4
    vpaddd      %ymm4,  %ymm2,  %ymm2

    vpmaddwd    %ymm10, %ymm14, %ymm4
    vpaddd      %ymm4,  %ymm3,  %ymm3

    addq    $ 16*SIZE, AO1
    addq    $ 16*SIZE, AO2
    addq    $ 16*SIZE, BO1
    addq    $ 16*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL2x2_8_SUB

    movdqu     0 * SIZE(AO1), %xmm8
    movdqu     0 * SIZE(AO2), %xmm10

    movdqu     0 * SIZE(BO1), %xmm12
    movdqu     0 * SIZE(BO2), %xmm14

    movdqu     %xmm8,  %xmm4

    pmaddwd    %xmm12,  %xmm8
    paddd      %xmm8,   %xmm0

    pmaddwd    %xmm14,  %xmm4
    paddd      %xmm4,   %xmm1

    movdqu     %xmm10,  %xmm4

    pmaddwd    %xmm12, %xmm10
    paddd      %xmm10,  %xmm2

    pmaddwd    %xmm14,  %xmm4
    paddd      %xmm4,   %xmm3

    addq    $ 8*SIZE, AO1
    addq    $ 8*SIZE, AO2
    addq    $ 8*SIZE, BO1
    addq    $ 8*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL2x2_4_SUB

    movq       0 * SIZE(AO1), %xmm8
    movq       0 * SIZE(AO2), %xmm10

    movq       0 * SIZE(BO1), %xmm12
    movq       0 * SIZE(BO2), %xmm14

    movdqu     %xmm8,  %xmm4

    pmaddwd    %xmm12,  %xmm8
    paddd      %xmm8,   %xmm0

    pmaddwd    %xmm14,  %xmm4
    paddd      %xmm4,   %xmm1

    movdqu     %xmm10,  %xmm4

    pmaddwd    %xmm12, %xmm10
    paddd      %xmm10,  %xmm2

    pmaddwd    %xmm14,  %xmm4
    paddd      %xmm4,   %xmm3

    addq    $ 4*SIZE, AO1
    addq    $ 4*SIZE, AO2
    addq    $ 4*SIZE, BO1
    addq    $ 4*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL2x2_2_SUB

    movd       0 * SIZE(AO1), %xmm8
    movd       0 * SIZE(AO2), %xmm10

    movd       0 * SIZE(BO1), %xmm12
    movd       0 * SIZE(BO2), %xmm14

    movdqu     %xmm8,  %xmm4

    pmaddwd    %xmm12,  %xmm8
    paddd      %xmm8,   %xmm0

    pmaddwd    %xmm14,  %xmm4
    paddd      %xmm4,   %xmm1

    movdqu     %xmm10,  %xmm4

    pmaddwd    %xmm12, %xmm10
    paddd      %xmm10,  %xmm2

    pmaddwd    %xmm14,  %xmm4
    paddd      %xmm4,   %xmm3

    addq    $ 2*SIZE, AO1
    addq    $ 2*SIZE, AO2
    addq    $ 2*SIZE, BO1
    addq    $ 2*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL2x2_1_SUB

    movzwl     0 * SIZE(AO1), TEMP_D
    movd       TEMP_D, %xmm8
    movzwl     0 * SIZE(AO2), TEMP_D
    movd       TEMP_D, %xmm10

    movzwl     0 * SIZE(BO1), TEMP_D
    movd       TEMP_D, %xmm12
    movzwl     0 * SIZE(BO2), TEMP_D
    movd       TEMP_D, %xmm14

    movdqu     %xmm8,  %xmm4

    pmaddwd    %xmm12,  %xmm8
    paddd      %xmm8,   %xmm0

    pmaddwd    %xmm14,  %xmm4
    paddd      %xmm4,   %xmm1

    movdqu     %xmm10,  %xmm4

    pmaddwd    %xmm12, %xmm10
    paddd      %xmm10,  %xmm2

    pmaddwd    %xmm14,  %xmm4
    paddd      %xmm4,   %xmm3

    addq    $ 1*SIZE, AO1
    addq    $ 1*SIZE, AO2
    addq    $ 1*SIZE, BO1
    addq    $ 1*SIZE, BO2

.endm

/*******************************************************************************************/

.macro SAVE2x2_SUB

    vextractf128 $ 1, %ymm0, %xmm8     # %ymm0 hi 128bit --> %xmm1
    paddd     %xmm8, %xmm0             # %ymm0 = %ymm1 + %ymm0
    phaddd    %xmm0, %xmm0             # xmm0[0] = xmm0[0] + xmm0[1], xmm0[1] = xmm0[2] + xmm0[3]
    phaddd    %xmm0, %xmm0
    movq      %xmm0, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(CO1)

    vextractf128 $ 1, %ymm1, %xmm8
    paddd     %xmm8, %xmm1
    phaddd    %xmm1, %xmm1
    phaddd    %xmm1, %xmm1
    movq      %xmm1, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 1 * SIZE(CO1)

    vextractf128 $ 1, %ymm2, %xmm8
    paddd     %xmm8, %xmm2
    phaddd    %xmm2, %xmm2
    phaddd    %xmm2, %xmm2
    movq      %xmm2, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(CO2)

    vextractf128 $ 1, %ymm3, %xmm8
    paddd     %xmm8, %xmm3
    phaddd    %xmm3, %xmm3
    phaddd    %xmm3, %xmm3
    movq      %xmm3, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 1 * SIZE(CO2)

    addq      $ 2*SIZE, CO1
    addq      $ 2*SIZE, CO2

.endm

/*******************************************************************************************/

.macro RESET2x2_SUB

    vxorps  %ymm0, %ymm0, %ymm0 # %ymm0 = 0
    vxorps  %ymm1, %ymm1, %ymm1 # %ymm1 = 0
    vxorps  %ymm2, %ymm2, %ymm2 # %ymm2 = 0
    vxorps  %ymm3, %ymm3, %ymm3 # %ymm3 = 0

.endm

/*******************************************************************************************/

.macro KERNEL2x1_32_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8
    vmovdqu    16 * SIZE(AO1), %ymm9
    vmovdqu     0 * SIZE(AO2), %ymm10
    vmovdqu    16 * SIZE(AO2), %ymm11

    vmovdqu     0 * SIZE(BO1), %ymm12
    vmovdqu    16 * SIZE(BO1), %ymm13

    vpmaddwd    %ymm8,  %ymm12, %ymm4
    vpmaddwd    %ymm9,  %ymm13, %ymm5
    vpaddd      %ymm4,  %ymm5,  %ymm4
    vpaddd      %ymm4,  %ymm0,  %ymm0

    vpmaddwd    %ymm10, %ymm12, %ymm4
    vpmaddwd    %ymm11, %ymm13, %ymm5
    vpaddd      %ymm4,  %ymm5,  %ymm4
    vpaddd      %ymm4,  %ymm1,  %ymm1

    addq    $ 32*SIZE, AO1
    addq    $ 32*SIZE, AO2
    addq    $ 32*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL2x1_16_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8
    vmovdqu     0 * SIZE(AO2), %ymm10

    vmovdqu     0 * SIZE(BO1), %ymm12

    vpmaddwd    %ymm8,  %ymm12, %ymm4
    vpaddd      %ymm4,  %ymm0,  %ymm0

    vpmaddwd    %ymm10, %ymm12, %ymm4
    vpaddd      %ymm4,  %ymm1,  %ymm1

    addq    $ 16*SIZE, AO1
    addq    $ 16*SIZE, AO2
    addq    $ 16*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL2x1_8_SUB

    movdqu     0 * SIZE(AO1), %xmm8
    movdqu     0 * SIZE(AO2), %xmm10

    movdqu     0 * SIZE(BO1), %xmm12

    pmaddwd    %xmm12,  %xmm8
    paddd      %xmm8,   %xmm0

    pmaddwd    %xmm12, %xmm10
    paddd      %xmm10,  %xmm1

    addq    $ 8*SIZE, AO1
    addq    $ 8*SIZE, AO2
    addq    $ 8*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL2x1_4_SUB

    movq       0 * SIZE(AO1), %xmm8
    movq       0 * SIZE(AO2), %xmm10

    movq       0 * SIZE(BO1), %xmm12

    pmaddwd    %xmm12,  %xmm8
    paddd      %xmm8,   %xmm0

    pmaddwd    %xmm12, %xmm10
    paddd      %xmm10,  %xmm1

    addq    $ 4*SIZE, AO1
    addq    $ 4*SIZE, AO2
    addq    $ 4*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL2x1_2_SUB

    movd       0 * SIZE(AO1), %xmm8
    movd       0 * SIZE(AO2), %xmm10

    movd       0 * SIZE(BO1), %xmm12

    pmaddwd    %xmm12,  %xmm8
    paddd      %xmm8,   %xmm0

    pmaddwd    %xmm12, %xmm10
    paddd      %xmm10,  %xmm1

    addq    $ 2*SIZE, AO1
    addq    $ 2*SIZE, AO2
    addq    $ 2*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL2x1_1_SUB

    movzwl     0 * SIZE(AO1), TEMP_D
    movd       TEMP_D, %xmm8
    movzwl     0 * SIZE(AO2), TEMP_D
    movd       TEMP_D, %xmm10

    movzwl     0 * SIZE(BO1), TEMP_D
    movd       TEMP_D, %xmm12

    pmaddwd    %xmm12,  %xmm8
    paddd      %xmm8,   %xmm0

    pmaddwd    %xmm12, %xmm10
    paddd      %xmm10,  %xmm1

    addq    $ 1*SIZE, AO1
    addq    $ 1*SIZE, AO2
    addq    $ 1*SIZE, BO1

.endm

/*******************************************************************************************/

.macro SAVE2x1_SUB

    vextractf128 $ 1, %ymm0, %xmm8     # %ymm0 hi 128bit --> %xmm1
    paddd     %xmm8, %xmm0             # %ymm0 = %ymm1 + %ymm0
    phaddd    %xmm0, %xmm0             # xmm0[0] = xmm0[0] + xmm0[1], xmm0[1] = xmm0[2] + xmm0[3]
    phaddd    %xmm0, %xmm0
    movq      %xmm0, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(CO1)

    vextractf128 $ 1, %ymm1, %xmm8
    paddd     %xmm8, %xmm1
    phaddd    %xmm1, %xmm1
    phaddd    %xmm1, %xmm1
    movq      %xmm1, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(CO2)

    addq      $ 1*SIZE, CO1
    addq      $ 1*SIZE, CO2

.endm

/*******************************************************************************************/

.macro RESET2x1_SUB

    vxorps  %ymm0, %ymm0, %ymm0 # %ymm0 = 0
    vxorps  %ymm1, %ymm1, %ymm1 # %ymm1 = 0

.endm

/*******************************************************************************************/

.macro KERNEL1x4_32_SUB

    vmovdqu     0 * SIZE(AO1), %ymm6
    vmovdqu    16 * SIZE(AO1), %ymm7

    vmovdqu     0 * SIZE(BO1), %ymm8
    vmovdqu    16 * SIZE(BO1), %ymm9
    vmovdqu     0 * SIZE(BO2), %ymm10
    vmovdqu    16 * SIZE(BO2), %ymm11
    vmovdqu     0 * SIZE(BO3), %ymm12
    vmovdqu    16 * SIZE(BO3), %ymm13
    vmovdqu     0 * SIZE(BO4), %ymm14
    vmovdqu    16 * SIZE(BO4), %ymm15

    vpmaddwd    %ymm6, %ymm8,  %ymm4
    vpmaddwd    %ymm7, %ymm9,  %ymm5
    vpaddd      %ymm4, %ymm5,  %ymm4
    vpaddd      %ymm4, %ymm0,  %ymm0

    vpmaddwd    %ymm6, %ymm10, %ymm4
    vpmaddwd    %ymm7, %ymm11, %ymm5
    vpaddd      %ymm4, %ymm5,  %ymm4
    vpaddd      %ymm4, %ymm1,  %ymm1

    vpmaddwd    %ymm6, %ymm12, %ymm4
    vpmaddwd    %ymm7, %ymm13, %ymm5
    vpaddd      %ymm4, %ymm5,  %ymm4
    vpaddd      %ymm4, %ymm2,  %ymm2

    vpmaddwd    %ymm6, %ymm14, %ymm4
    vpmaddwd    %ymm7, %ymm15, %ymm5
    vpaddd      %ymm4, %ymm5,  %ymm4
    vpaddd      %ymm4, %ymm3,  %ymm3

    addq    $ 32*SIZE, AO1
    addq    $ 32*SIZE, BO1
    addq    $ 32*SIZE, BO2
    addq    $ 32*SIZE, BO3
    addq    $ 32*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL1x4_16_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8

    vmovdqu     0 * SIZE(BO1), %ymm12
    vmovdqu     0 * SIZE(BO2), %ymm13
    vmovdqu     0 * SIZE(BO3), %ymm14
    vmovdqu     0 * SIZE(BO4), %ymm15

    vpmaddwd    %ymm8, %ymm12, %ymm12
    vpmaddwd    %ymm8, %ymm13, %ymm13
    vpmaddwd    %ymm8, %ymm14, %ymm14
    vpmaddwd    %ymm8, %ymm15, %ymm15

    vpaddd      %ymm12, %ymm0, %ymm0
    vpaddd      %ymm13, %ymm1, %ymm1
    vpaddd      %ymm14, %ymm2, %ymm2
    vpaddd      %ymm15, %ymm3, %ymm3

    addq    $ 16*SIZE, AO1
    addq    $ 16*SIZE, BO1
    addq    $ 16*SIZE, BO2
    addq    $ 16*SIZE, BO3
    addq    $ 16*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL1x4_8_SUB

    movdqu     0 * SIZE(AO1), %xmm8

    movdqu     0 * SIZE(BO1), %xmm12
    movdqu     0 * SIZE(BO2), %xmm13
    movdqu     0 * SIZE(BO3), %xmm14
    movdqu     0 * SIZE(BO4), %xmm15

    pmaddwd    %xmm8, %xmm12
    pmaddwd    %xmm8, %xmm13
    pmaddwd    %xmm8, %xmm14
    pmaddwd    %xmm8, %xmm15

    paddd      %xmm12, %xmm0
    paddd      %xmm13, %xmm1
    paddd      %xmm14, %xmm2
    paddd      %xmm15, %xmm3

    addq    $ 8*SIZE, AO1
    addq    $ 8*SIZE, BO1
    addq    $ 8*SIZE, BO2
    addq    $ 8*SIZE, BO3
    addq    $ 8*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL1x4_4_SUB

    movq     0 * SIZE(AO1), %xmm8

    movq     0 * SIZE(BO1), %xmm12
    movq     0 * SIZE(BO2), %xmm13
    movq     0 * SIZE(BO3), %xmm14
    movq     0 * SIZE(BO4), %xmm15

    pmaddwd    %xmm8, %xmm12
    pmaddwd    %xmm8, %xmm13
    pmaddwd    %xmm8, %xmm14
    pmaddwd    %xmm8, %xmm15

    paddd      %xmm12, %xmm0
    paddd      %xmm13, %xmm1
    paddd      %xmm14, %xmm2
    paddd      %xmm15, %xmm3

    addq    $ 4*SIZE, AO1
    addq    $ 4*SIZE, BO1
    addq    $ 4*SIZE, BO2
    addq    $ 4*SIZE, BO3
    addq    $ 4*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL1x4_2_SUB

    movd     0 * SIZE(AO1), %xmm8

    movd     0 * SIZE(BO1), %xmm12
    movd     0 * SIZE(BO2), %xmm13
    movd     0 * SIZE(BO3), %xmm14
    movd     0 * SIZE(BO4), %xmm15

    pmaddwd    %xmm8, %xmm12
    pmaddwd    %xmm8, %xmm13
    pmaddwd    %xmm8, %xmm14
    pmaddwd    %xmm8, %xmm15

    paddd      %xmm12, %xmm0
    paddd      %xmm13, %xmm1
    paddd      %xmm14, %xmm2
    paddd      %xmm15, %xmm3

    addq    $ 2*SIZE, AO1
    addq    $ 2*SIZE, BO1
    addq    $ 2*SIZE, BO2
    addq    $ 2*SIZE, BO3
    addq    $ 2*SIZE, BO4

.endm

/*******************************************************************************************/

.macro KERNEL1x4_1_SUB

    movzwl    0 * SIZE(AO1), TEMP_D
    movd      TEMP_D, %xmm8

    movzwl    0 * SIZE(BO1), TEMP_D
    movd      TEMP_D, %xmm12
    movzwl    0 * SIZE(BO2), TEMP_D
    movd      TEMP_D, %xmm13
    movzwl    0 * SIZE(BO3), TEMP_D
    movd      TEMP_D, %xmm14
    movzwl    0 * SIZE(BO4), TEMP_D
    movd      TEMP_D, %xmm15

    pmaddwd    %xmm8, %xmm12
    pmaddwd    %xmm8, %xmm13
    pmaddwd    %xmm8, %xmm14
    pmaddwd    %xmm8, %xmm15

    paddd      %xmm12, %xmm0
    paddd      %xmm13, %xmm1
    paddd      %xmm14, %xmm2
    paddd      %xmm15, %xmm3

    addq    $ 1*SIZE, AO1
    addq    $ 1*SIZE, BO1
    addq    $ 1*SIZE, BO2
    addq    $ 1*SIZE, BO3
    addq    $ 1*SIZE, BO4

.endm

/*******************************************************************************************/

.macro SAVE1x4_SUB

    vextractf128 $ 1, %ymm0, %xmm8     # %ymm0 hi 128bit --> %xmm1
    paddd     %xmm8, %xmm0             # %ymm0 = %ymm1 + %ymm0
    phaddd    %xmm0, %xmm0             # xmm0[0] = xmm0[0] + xmm0[1], xmm0[1] = xmm0[2] + xmm0[3]
    phaddd    %xmm0, %xmm0
    movq      %xmm0, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(CO1)

    vextractf128 $ 1, %ymm1, %xmm8
    paddd     %xmm8, %xmm1
    phaddd    %xmm1, %xmm1
    phaddd    %xmm1, %xmm1
    movq      %xmm1, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 1 * SIZE(CO1)

    vextractf128 $ 1, %ymm2, %xmm8
    paddd     %xmm8, %xmm2
    phaddd    %xmm2, %xmm2
    phaddd    %xmm2, %xmm2
    movq      %xmm2, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 2 * SIZE(CO1)

    vextractf128 $ 1, %ymm3, %xmm8
    paddd     %xmm8, %xmm3
    phaddd    %xmm3, %xmm3
    phaddd    %xmm3, %xmm3
    movq      %xmm3, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 3 * SIZE(CO1)

    addq      $ 4*SIZE, CO1

.endm

/*******************************************************************************************/

.macro RESET1x4_SUB

    vxorps  %ymm0, %ymm0, %ymm0 # %ymm0 = 0
    vxorps  %ymm1, %ymm1, %ymm1 # %ymm1 = 0
    vxorps  %ymm2, %ymm2, %ymm2 # %ymm2 = 0
    vxorps  %ymm3, %ymm3, %ymm3 # %ymm3 = 0

.endm

/*******************************************************************************************/

.macro KERNEL1x2_32_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8
    vmovdqu    16 * SIZE(AO1), %ymm9

    vmovdqu     0 * SIZE(BO1), %ymm12
    vmovdqu    16 * SIZE(BO1), %ymm13
    vmovdqu     0 * SIZE(BO2), %ymm14
    vmovdqu    16 * SIZE(BO2), %ymm15

    vpmaddwd    %ymm8,  %ymm12, %ymm4
    vpmaddwd    %ymm9,  %ymm13, %ymm5
    vpaddd      %ymm4,  %ymm5,  %ymm4
    vpaddd      %ymm4,  %ymm0,  %ymm0

    vpmaddwd    %ymm8,  %ymm14, %ymm4
    vpmaddwd    %ymm9,  %ymm15, %ymm5
    vpaddd      %ymm4,  %ymm5,  %ymm4
    vpaddd      %ymm4,  %ymm1,  %ymm1

    addq    $ 32*SIZE, AO1
    addq    $ 32*SIZE, BO1
    addq    $ 32*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL1x2_16_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8

    vmovdqu     0 * SIZE(BO1), %ymm12
    vmovdqu     0 * SIZE(BO2), %ymm14

    vpmaddwd    %ymm8,  %ymm12, %ymm4
    vpaddd      %ymm4,  %ymm0,  %ymm0

    vpmaddwd    %ymm8,  %ymm14, %ymm4
    vpaddd      %ymm4,  %ymm1,  %ymm1

    addq    $ 16*SIZE, AO1
    addq    $ 16*SIZE, BO1
    addq    $ 16*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL1x2_8_SUB

    movdqu     0 * SIZE(AO1), %xmm8

    movdqu     0 * SIZE(BO1), %xmm12
    movdqu     0 * SIZE(BO2), %xmm14

    pmaddwd    %xmm8,  %xmm12
    paddd      %xmm12,  %xmm0

    pmaddwd    %xmm8,  %xmm14
    paddd      %xmm14,  %xmm1

    addq    $ 8*SIZE, AO1
    addq    $ 8*SIZE, BO1
    addq    $ 8*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL1x2_4_SUB

    movq       0 * SIZE(AO1), %xmm8

    movq       0 * SIZE(BO1), %xmm12
    movq       0 * SIZE(BO2), %xmm14

    pmaddwd    %xmm8,  %xmm12
    paddd      %xmm12,  %xmm0

    pmaddwd    %xmm8,  %xmm14
    paddd      %xmm14,  %xmm1

    addq    $ 4*SIZE, AO1
    addq    $ 4*SIZE, BO1
    addq    $ 4*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL1x2_2_SUB

    movd       0 * SIZE(AO1), %xmm8

    movd       0 * SIZE(BO1), %xmm12
    movd       0 * SIZE(BO2), %xmm14

    pmaddwd    %xmm8,  %xmm12
    paddd      %xmm12,  %xmm0

    pmaddwd    %xmm8,  %xmm14
    paddd      %xmm14,  %xmm1

    addq    $ 2*SIZE, AO1
    addq    $ 2*SIZE, BO1
    addq    $ 2*SIZE, BO2

.endm

/*******************************************************************************************/

.macro KERNEL1x2_1_SUB

    movzwl     0 * SIZE(AO1), TEMP_D
    movd       TEMP_D, %xmm8

    movzwl     0 * SIZE(BO1), TEMP_D
    movd       TEMP_D, %xmm12
    movzwl     0 * SIZE(BO2), TEMP_D
    movd       TEMP_D, %xmm14

    pmaddwd    %xmm8,  %xmm12
    paddd      %xmm12,  %xmm0

    pmaddwd    %xmm8,  %xmm14
    paddd      %xmm14,  %xmm1

    addq    $ 1*SIZE, AO1
    addq    $ 1*SIZE, BO1
    addq    $ 1*SIZE, BO2

.endm

/*******************************************************************************************/

.macro SAVE1x2_SUB

    vextractf128 $ 1, %ymm0, %xmm8     # %ymm0 hi 128bit --> %xmm1
    paddd     %xmm8, %xmm0             # %ymm0 = %ymm1 + %ymm0
    phaddd    %xmm0, %xmm0             # xmm0[0] = xmm0[0] + xmm0[1], xmm0[1] = xmm0[2] + xmm0[3]
    phaddd    %xmm0, %xmm0
    movq      %xmm0, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(CO1)

    vextractf128 $ 1, %ymm1, %xmm8
    paddd     %xmm8, %xmm1
    phaddd    %xmm1, %xmm1
    phaddd    %xmm1, %xmm1
    movq      %xmm1, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 1 * SIZE(CO1)

    addq      $ 2*SIZE, CO1

.endm

/*******************************************************************************************/

.macro RESET1x2_SUB

    vxorps  %ymm0, %ymm0, %ymm0 # %ymm0 = 0
    vxorps  %ymm1, %ymm1, %ymm1 # %ymm1 = 0

.endm

/*******************************************************************************************/

.macro KERNEL1x1_32_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8
    vmovdqu    16 * SIZE(AO1), %ymm9

    vmovdqu     0 * SIZE(BO1), %ymm12
    vmovdqu    16 * SIZE(BO1), %ymm13

    vpmaddwd    %ymm8,  %ymm12, %ymm4
    vpmaddwd    %ymm9,  %ymm13, %ymm5
    vpaddd      %ymm4,  %ymm5,  %ymm4
    vpaddd      %ymm4,  %ymm0,  %ymm0

    addq    $ 32*SIZE, AO1
    addq    $ 32*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL1x1_16_SUB

    vmovdqu     0 * SIZE(AO1), %ymm8

    vmovdqu     0 * SIZE(BO1), %ymm12

    vpmaddwd    %ymm8,  %ymm12, %ymm4
    vpaddd      %ymm4,  %ymm0,  %ymm0

    addq    $ 16*SIZE, AO1
    addq    $ 16*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL1x1_8_SUB

    movdqu     0 * SIZE(AO1), %xmm8

    movdqu     0 * SIZE(BO1), %xmm12

    pmaddwd    %xmm8,  %xmm12
    paddd      %xmm12,  %xmm0

    addq    $ 8*SIZE, AO1
    addq    $ 8*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL1x1_4_SUB

    movq       0 * SIZE(AO1), %xmm8

    movq       0 * SIZE(BO1), %xmm12

    pmaddwd    %xmm8,  %xmm12
    paddd      %xmm12,  %xmm0

    addq    $ 4*SIZE, AO1
    addq    $ 4*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL1x1_2_SUB

    movd       0 * SIZE(AO1), %xmm8

    movd       0 * SIZE(BO1), %xmm12

    pmaddwd    %xmm8,  %xmm12
    paddd      %xmm12,  %xmm0

    addq    $ 2*SIZE, AO1
    addq    $ 2*SIZE, BO1

.endm

/*******************************************************************************************/

.macro KERNEL1x1_1_SUB

    movzwl     0 * SIZE(AO1), TEMP_D
    movd       TEMP_D, %xmm8

    movzwl     0 * SIZE(BO1), TEMP_D
    movd       TEMP_D, %xmm12

    pmaddwd    %xmm8,  %xmm12
    paddd      %xmm12,  %xmm0

    addq    $ 1*SIZE, AO1
    addq    $ 1*SIZE, BO1

.endm

/*******************************************************************************************/

.macro SAVE1x1_SUB

    vextractf128 $ 1, %ymm0, %xmm8     # %ymm0 hi 128bit --> %xmm1
    paddd     %xmm8, %xmm0             # %ymm0 = %ymm1 + %ymm0
    phaddd    %xmm0, %xmm0             # xmm0[0] = xmm0[0] + xmm0[1], xmm0[1] = xmm0[2] + xmm0[3]
    phaddd    %xmm0, %xmm0
    movq      %xmm0, RESULT
    sar       ALPHA_B, RESULT
    add       RESULT_W, 0 * SIZE(CO1)

.endm

/*******************************************************************************************/

.macro RESET1x1_SUB

    vxorps  %ymm0, %ymm0, %ymm0 # %ymm0 = 0

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

    salq    $BASE_SHIFT, LDA
    salq    $BASE_SHIFT, LDB
    salq    $BASE_SHIFT, LDC

.L0_0:
    cmpq    $2, M
    jl      .L0_2

    push    M
    push    SP

    movq    A,  AO1
    leaq    (A, LDA, 1), AO2
    leaq    (A, LDA, 2), A      # A = A + 2*LDA
    movq    C,  CO1
    leaq    (C, LDC, 1), CO2
    leaq    (C, LDC, 2), C      # C = C + 2*LDC

    push    A                   # svae registers
    push    LDA
    push    C
    push    LDC

    movq    B, BO1
    leaq    (B, LDB, 1),   BO2
    leaq    (B, LDB, 2),   BO3
    leaq    (BO3, LDB, 1), BO4

    push    B

    movq    N,  J

.L1_0:
    cmpq    $4, J
    jl     .L1_1

    # A(2) x B(4)
    movq  K, KK
    RESET2x4_SUB

.L2_2x4_0:
    cmpq  $32, KK
    jl .L2_2x4_1

    KERNEL2x4_32_SUB

    ALIGN_4
    subq  $32, KK
    jnz .L2_2x4_0

.L2_2x4_1:
    testq $16, KK
    jz .L2_2x4_2

    KERNEL2x4_16_SUB

    ALIGN_4

.L2_2x4_2:
    testq $8, KK
    jz .L2_2x4_3

    KERNEL2x4_8_SUB

    ALIGN_4

.L2_2x4_3:
    testq $4, KK
    jz .L2_2x4_4

    KERNEL2x4_4_SUB

    ALIGN_4

.L2_2x4_4:
    testq $2, KK
    jz .L2_2x4_5

    KERNEL2x4_2_SUB

    ALIGN_4

.L2_2x4_5:
    testq $1, KK
    jz .L2_2x4_6

    KERNEL2x4_1_SUB

    ALIGN_4

.L2_2x4_6:

    SAVE2x4_SUB

    ALIGN_4

    # reset AO1, AO2
    neg     K
    leaq    (AO1, K, SIZE), AO1
    leaq    (AO2, K, SIZE), AO2

    # set BO1, BO2, BO3, BO4
    leaq    (BO1, K, SIZE), BO1
    leaq    (BO2, K, SIZE), BO2
    leaq    (BO3, K, SIZE), BO3
    leaq    (BO4, K, SIZE), BO4

    leaq    (BO1, LDB, 4), BO1
    leaq    (BO2, LDB, 4), BO2
    leaq    (BO3, LDB, 4), BO3
    leaq    (BO4, LDB, 4), BO4
    neg     K

    subq    $4, J
    jnz     .L1_0

.L1_1:
    testq   $2, J
    jz      .L1_2

    # A(2) x B(2)
    movq  K, KK
    RESET2x2_SUB

.L2_2x2_0:
    cmpq  $32, KK
    jl .L2_2x2_1

    KERNEL2x2_32_SUB

    ALIGN_4
    subq  $32, KK
    jnz .L2_2x2_0

.L2_2x2_1:
    testq $16, KK
    jz .L2_2x2_2

    KERNEL2x2_16_SUB

    ALIGN_4

.L2_2x2_2:
    testq $8, KK
    jz .L2_2x2_3

    KERNEL2x2_8_SUB

    ALIGN_4

.L2_2x2_3:
    testq $4, KK
    jz .L2_2x2_4

    KERNEL2x2_4_SUB

    ALIGN_4

.L2_2x2_4:
    test $2, KK
    jz .L2_2x2_5

    KERNEL2x2_2_SUB

    ALIGN_4

.L2_2x2_5:
    test $1, KK
    jz .L2_2x2_6

    KERNEL2x2_1_SUB

    ALIGN_4

.L2_2x2_6:

    SAVE2x2_SUB

    ALIGN_4

    # reset AO1, AO2
    neg     K
    leaq    (AO1, K, SIZE), AO1
    leaq    (AO2, K, SIZE), AO2

    # set BO1, BO2
    leaq    (BO1, K, SIZE), BO1
    leaq    (BO2, K, SIZE), BO2

    leaq    (BO1, LDB, 2), BO1
    leaq    (BO2, LDB, 2), BO2
    neg     K

.L1_2:
    testq   $1, J
    jz      .L0_1

    # A(2) x B(1)
    movq  K, KK
    RESET2x1_SUB

.L2_2x1_0:
    cmpq  $32, KK
    jl .L2_2x1_1

    KERNEL2x1_32_SUB

    ALIGN_4
    subq  $32, KK
    jnz .L2_2x1_0

.L2_2x1_1:
    testq $16, KK
    jz .L2_2x1_2

    KERNEL2x1_16_SUB

    ALIGN_4

.L2_2x1_2:
    testq $8, KK
    jz .L2_2x1_3

    KERNEL2x1_8_SUB

    ALIGN_4

.L2_2x1_3:
    testq $4, KK
    jz .L2_2x1_4

    KERNEL2x1_4_SUB

    ALIGN_4

.L2_2x1_4:
    testq $2, KK
    jz .L2_2x1_5

    KERNEL2x1_2_SUB

    ALIGN_4

.L2_2x1_5:
    testq $1, KK
    jz .L2_2x1_6

    KERNEL2x1_1_SUB

    ALIGN_4

.L2_2x1_6:

    SAVE2x1_SUB

    ALIGN_4

.L0_1:
    pop     B
    pop     LDC
    pop     C
    pop     LDA
    pop     A
    pop     SP
    pop     M
    subq    $2, M
    jnz     .L0_0

.L0_2:
    testq   $1, M                # test rest of M
    jz      .L999

    movq    A,  AO1
    movq    C,  CO1

    movq    B, BO1
    leaq    (B, LDB, 1),   BO2
    leaq    (B, LDB, 2),   BO3
    leaq    (BO3, LDB, 1), BO4

    movq    N,  J

.L1_3:
    cmpq    $4, J
    jl      .L1_4

    # A(1) x B(4)
    movq  K, KK
    RESET1x4_SUB

.L2_1x4_0:
    cmpq  $32, KK
    jl  .L2_1x4_1

    KERNEL1x4_32_SUB

    ALIGN_4
    subq  $32, KK
    jnz .L2_1x4_0

.L2_1x4_1:
    testq $16, KK
    jz .L2_1x4_2

    KERNEL1x4_16_SUB

    ALIGN_4

.L2_1x4_2:
    testq $8, KK
    jz .L2_1x4_3

    KERNEL1x4_8_SUB

    ALIGN_4

.L2_1x4_3:
    testq $4, KK
    jz .L2_1x4_4

    KERNEL1x4_4_SUB

    ALIGN_4

.L2_1x4_4:
    testq $2, KK
    jz .L2_1x4_5

    KERNEL1x4_2_SUB

    ALIGN_4

.L2_1x4_5:
    testq $1, KK
    jz .L2_1x4_6

    KERNEL1x4_1_SUB

    ALIGN_4

.L2_1x4_6:

    SAVE1x4_SUB

    ALIGN_4

    # reset AO1, AO2
    neg     K
    leaq    (AO1, K, SIZE), AO1

    # set BO1, BO2, BO3, BO4
    leaq    (BO1, K, SIZE), BO1
    leaq    (BO2, K, SIZE), BO2
    leaq    (BO3, K, SIZE), BO3
    leaq    (BO4, K, SIZE), BO4

    leaq    (BO1, LDB, 4), BO1
    leaq    (BO2, LDB, 4), BO2
    leaq    (BO3, LDB, 4), BO3
    leaq    (BO4, LDB, 4), BO4
    neg     K

    subq    $4, J
    jnz     .L1_3

.L1_4:

    testq   $2, J
    jz      .L1_5

    # A(1) x B(2)
    movq  K, KK
    RESET1x2_SUB

.L2_1x2_0:
    cmpq  $32, KK
    jl    .L2_1x2_1

    KERNEL1x2_32_SUB

    ALIGN_4
    subq  $32, KK
    jnz .L2_1x2_0

.L2_1x2_1:
    testq $16, KK
    jz .L2_1x2_2

    KERNEL1x2_16_SUB

    ALIGN_4

.L2_1x2_2:
    testq $8, KK
    jz .L2_1x2_3

    KERNEL1x2_8_SUB

    ALIGN_4

.L2_1x2_3:
    testq $4, KK
    jz .L2_1x2_4

    KERNEL1x2_4_SUB

    ALIGN_4

.L2_1x2_4:
    testq $2, KK
    jz .L2_1x2_5

    KERNEL1x2_2_SUB

    ALIGN_4

.L2_1x2_5:
    testq $1, KK
    jz .L2_1x2_6

    KERNEL1x2_1_SUB

    ALIGN_4

.L2_1x2_6:

    SAVE1x2_SUB

    ALIGN_4

    # reset AO1, AO2
    neg     K
    leaq    (AO1, K, SIZE), AO1

    # set BO1, BO2, BO3, BO4
    leaq    (BO1, K, SIZE), BO1
    leaq    (BO2, K, SIZE), BO2

    leaq    (BO1, LDB, 2), BO1
    leaq    (BO2, LDB, 2), BO2
    neg     K

.L1_5:
    testq   $1, J
    jz      .L999

    # A(1) x B(1)
    movq  K, KK
    RESET1x1_SUB

.L2_1x1_0:
    cmpq  $32, KK
    jl    .L2_1x1_1

    KERNEL1x1_32_SUB

    ALIGN_4
    subq  $32, KK
    jnz .L2_1x1_0

.L2_1x1_1:
    testq $16, KK
    jz .L2_1x1_2

    KERNEL1x1_16_SUB

    ALIGN_4

.L2_1x1_2:
    testq $8, KK
    jz .L2_1x1_3

    KERNEL1x1_8_SUB

    ALIGN_4

.L2_1x1_3:
    testq $4, KK
    jz .L2_1x1_4

    KERNEL1x1_4_SUB

    ALIGN_4

.L2_1x1_4:
    testq $2, KK
    jz .L2_1x1_5

    KERNEL1x1_2_SUB

    ALIGN_4

.L2_1x1_5:
    testq $1, KK
    jz .L2_1x1_6

    KERNEL1x1_1_SUB

    ALIGN_4

.L2_1x1_6:

    SAVE1x1_SUB

    ALIGN_4

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
