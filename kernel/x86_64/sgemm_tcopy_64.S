/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#define ASSEMBLER
#include "common.h"

#if defined(PENTIUM4) || defined(GENERIC)
#define RPREFETCHSIZE    16
#define WPREFETCHSIZE (RPREFETCHSIZE * 4)
#define PREFETCH      prefetcht0
#define PREFETCHW     prefetcht0
#endif

#if defined(CORE2) || defined(PENRYN) || defined(DUNNINGTON) || defined(NEHALEM) || defined(SANDYBRIDGE)
#define RPREFETCHSIZE    12
#define WPREFETCHSIZE (RPREFETCHSIZE * 4)
#define PREFETCH      prefetcht0
#define PREFETCHW     prefetcht2
#endif

#ifdef ATOM
#define RPREFETCHSIZE    16
#define WPREFETCHSIZE (RPREFETCHSIZE * 4)
#define PREFETCH      prefetcht0
#define PREFETCHW     prefetcht0
#endif

#ifdef NANO
#define RPREFETCHSIZE    8
#define WPREFETCHSIZE (RPREFETCHSIZE * 4)
#define PREFETCH      prefetcht0
#define PREFETCHW     prefetcht0
#endif

#ifdef BARCELONA
#define RPREFETCHSIZE    8
#define WPREFETCHSIZE (RPREFETCHSIZE * 4)
#define PREFETCH      prefetch
#define PREFETCHW     prefetchw
#endif

#ifdef GENERIC
#define RPREFETCHSIZE    16
#define WPREFETCHSIZE (RPREFETCHSIZE * 4)
#define PREFETCH      prefetcht0
#define PREFETCHW     prefetcht0
#endif

#ifndef WINDOWS_ABI

#define M    ARG1    /* rdi */
#define N    ARG2    /* rsi */
#define A    ARG3    /* rdx */
#define LDA  ARG4    /* rcx */
#define B    ARG5    /* r8  */ /* ARG6 = 8 + STACKSIZE(%rsp) if exist, 8 = return address */ 

#define I    %rbp
#define J    %r9

#define AO    %r10
#define AO1   %r11
#define BO    %r12

#define LDA3  %r13
#define LDA5  %r14
#define LDA7  %r15


#else

#define STACKSIZE 256

#define M    ARG1    /* rcx */
#define N    ARG2    /* rdx */
#define A    ARG3    /* r8  */
#define LDA  ARG4    /* r9  */
#define OLD_B        40 + 64 + STACKSIZE(%rsp) /* 64 = push 8*8, 40 = return address(8) + ARG1-4(4*8) */

#define B    %rdi

#define I    %rsi
#define J    %rbp

#define AO    %r10
#define AO1   %r11
#define BO    %r12

#define LDA3  %r13
#define LDA5  %r14
#define LDA7  %r15

#endif

    PROLOGUE
    PROFCODE

#ifdef WINDOWS_ABI
    pushq    %rdi
    pushq    %rsi
#endif
    pushq    %r15
    pushq    %r14
    pushq    %r13
    pushq    %r12
    pushq    %rbp
    pushq    %rbx

#ifdef WINDOWS_ABI
    subq    $STACKSIZE, %rsp

    movups    %xmm6,    0(%rsp)
    movups    %xmm7,   16(%rsp)
    movups    %xmm8,   32(%rsp)
    movups    %xmm9,   48(%rsp)
    movups    %xmm10,  64(%rsp)
    movups    %xmm11,  80(%rsp)
    movups    %xmm12,  96(%rsp)
    movups    %xmm13, 112(%rsp)
    movups    %xmm14, 128(%rsp)
    movups    %xmm15, 144(%rsp)

    movq    OLD_B,     B
#endif

    movq    A,    AO
    movq    B,    BO

    leaq    (,LDA, SIZE), LDA
    leaq    (LDA,  LDA, 2), LDA3
    leaq    (LDA3, LDA, 2), LDA5
    leaq    (LDA5, LDA, 2), LDA7

    movq    N,    J
    sarq    $4,   J
    testq   J,    J
    jle .L20

.L10:
    movq    AO,   AO1
    addq    $16*SIZE, AO
    movq    M,    I
    sarq    $3,   I
    testq   I,    I
    jle .L12

    ALIGN_4

.L11:
    vmovups 0*SIZE(AO1), %ymm0
    vmovups 8*SIZE(AO1), %ymm1
    vmovups 0*SIZE(AO1, LDA), %ymm2
    vmovups 8*SIZE(AO1, LDA), %ymm3
    vmovups 0*SIZE(AO1, LDA, 2), %ymm4
    vmovups 8*SIZE(AO1, LDA, 2), %ymm5
    vmovups 0*SIZE(AO1, LDA3), %ymm6
    vmovups 8*SIZE(AO1, LDA3), %ymm7
    vmovups 0*SIZE(AO1, LDA, 4), %ymm8
    vmovups 8*SIZE(AO1, LDA, 4), %ymm9
    vmovups 0*SIZE(AO1, LDA5), %ymm10
    vmovups 8*SIZE(AO1, LDA5), %ymm11
    vmovups 0*SIZE(AO1, LDA3, 2), %ymm12
    vmovups 8*SIZE(AO1, LDA3, 2), %ymm13
    vmovups 0*SIZE(AO1, LDA7), %ymm14
    vmovups 8*SIZE(AO1, LDA7), %ymm15

    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)
    vmovups %ymm2, 16*SIZE(BO)
    vmovups %ymm3, 24*SIZE(BO)
    vmovups %ymm4, 32*SIZE(BO)
    vmovups %ymm5, 40*SIZE(BO)
    vmovups %ymm6, 48*SIZE(BO)
    vmovups %ymm7, 56*SIZE(BO)
    vmovups %ymm8, 64*SIZE(BO)
    vmovups %ymm9, 72*SIZE(BO)
    vmovups %ymm10, 80*SIZE(BO)
    vmovups %ymm11, 88*SIZE(BO)
    vmovups %ymm12, 96*SIZE(BO)
    vmovups %ymm13, 104*SIZE(BO)
    vmovups %ymm14, 112*SIZE(BO)
    vmovups %ymm15, 120*SIZE(BO)

    leaq (AO1, LDA, 8), AO1  # AO1 = AO1 + 8*LDA
    addq $128*SIZE, BO
    decq I
    testq I, I
    jg .L11

    ALIGN_4

.L12:
    testq $4, M
    jz .L13

    vmovups 0*SIZE(AO1), %ymm0
    vmovups 8*SIZE(AO1), %ymm1
    vmovups 0*SIZE(AO1, LDA), %ymm2
    vmovups 8*SIZE(AO1, LDA), %ymm3
    vmovups 0*SIZE(AO1, LDA, 2), %ymm4
    vmovups 8*SIZE(AO1, LDA, 2), %ymm5
    vmovups 0*SIZE(AO1, LDA3), %ymm6
    vmovups 8*SIZE(AO1, LDA3), %ymm7

    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)
    vmovups %ymm2, 16*SIZE(BO)
    vmovups %ymm3, 24*SIZE(BO)
    vmovups %ymm4, 32*SIZE(BO)
    vmovups %ymm5, 40*SIZE(BO)
    vmovups %ymm6, 48*SIZE(BO)
    vmovups %ymm7, 56*SIZE(BO)

    leaq (AO1, LDA, 4), AO1
    addq $64*SIZE, BO

    ALIGN_4

.L13:
    testq $2, M
    jz .L14

    vmovups 0*SIZE(AO1), %ymm0
    vmovups 8*SIZE(AO1), %ymm1
    vmovups 0*SIZE(AO1, LDA), %ymm2
    vmovups 8*SIZE(AO1, LDA), %ymm3

    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)
    vmovups %ymm2, 16*SIZE(BO)
    vmovups %ymm3, 24*SIZE(BO)

    leaq (AO1, LDA, 2), AO1
    addq $32*SIZE, BO

    ALIGN_4

.L14:
    testq $1, M
    jz .L15

    vmovups 0*SIZE(AO1), %ymm0
    vmovups 8*SIZE(AO1), %ymm1

    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)

    #leaq (AO1, LDA, 1), AO1
    addq $16*SIZE, BO

    ALIGN_4

.L15:
    decq J
    testq J, J
    jg .L10

.L20:
    testq $8, N
    jz .L30

    movq    AO,   AO1
    addq    $8*SIZE, AO
    movq    M,    I
    sarq    $3,   I
    testq   I,    I
    jle .L22

    ALIGN_4

.L21:
    vmovups (AO1), %ymm0
    vmovups (AO1, LDA), %ymm1
    vmovups (AO1, LDA, 2), %ymm2
    vmovups (AO1, LDA3), %ymm3
    vmovups (AO1, LDA, 4), %ymm4
    vmovups (AO1, LDA5), %ymm5
    vmovups (AO1, LDA3, 2), %ymm6
    vmovups (AO1, LDA7), %ymm7

    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)
    vmovups %ymm2, 16*SIZE(BO)
    vmovups %ymm3, 24*SIZE(BO)
    vmovups %ymm4, 32*SIZE(BO)
    vmovups %ymm5, 40*SIZE(BO)
    vmovups %ymm6, 48*SIZE(BO)
    vmovups %ymm7, 56*SIZE(BO)

    leaq (AO1, LDA, 8), AO1
    addq $64*SIZE, BO
    decq I
    testq I, I
    jg .L21

    ALIGN_4

.L22:
    testq $4, M
    jz .L23

    vmovups (AO1), %ymm0
    vmovups (AO1, LDA), %ymm1
    vmovups (AO1, LDA, 2), %ymm2
    vmovups (AO1, LDA3), %ymm3

    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)
    vmovups %ymm2, 16*SIZE(BO)
    vmovups %ymm3, 24*SIZE(BO)

    leaq (AO1, LDA, 4), AO1
    addq $32*SIZE, BO

    ALIGN_4

.L23:
    testq $2, M
    jz .L24

    vmovups (AO1), %ymm0
    vmovups (AO1, LDA), %ymm1

    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)

    leaq (AO1, LDA, 2), AO1
    addq $16*SIZE, BO

    ALIGN_4

.L24:
    testq $1, M
    jz .L30

    vmovups (AO1), %ymm0

    vmovups %ymm0, 0*SIZE(BO)

    #leaq (AO1, LDA, 1), AO1
    addq $8*SIZE, BO

    ALIGN_4

.L30:
    testq $4, N
    jz .L40

    movq    AO,   AO1
    addq    $4*SIZE, AO
    movq    M,    I
    sarq    $3,   I
    testq   I,    I
    jle .L32

    ALIGN_4

.L31:
    movups (AO1), %xmm0
    movups (AO1, LDA), %xmm1
    movups (AO1, LDA, 2), %xmm2
    movups (AO1, LDA3), %xmm3
    movups (AO1, LDA, 4), %xmm4
    movups (AO1, LDA5), %xmm5
    movups (AO1, LDA3, 2), %xmm6
    movups (AO1, LDA7), %xmm7

    movups %xmm0, 0*SIZE(BO)
    movups %xmm1, 4*SIZE(BO)
    movups %xmm2, 8*SIZE(BO)
    movups %xmm3, 12*SIZE(BO)
    movups %xmm4, 16*SIZE(BO)
    movups %xmm5, 20*SIZE(BO)
    movups %xmm6, 24*SIZE(BO)
    movups %xmm7, 28*SIZE(BO)

    leaq (AO1, LDA, 8), AO1
    addq $32*SIZE, BO
    decq I
    testq I, I
    jg .L31

    ALIGN_4

.L32:
    testq $4, M
    jz .L33

    movups (AO1), %xmm0
    movups (AO1, LDA), %xmm1
    movups (AO1, LDA, 2), %xmm2
    movups (AO1, LDA3), %xmm3

    movups %xmm0, 0*SIZE(BO)
    movups %xmm1, 4*SIZE(BO)
    movups %xmm2, 8*SIZE(BO)
    movups %xmm3, 12*SIZE(BO)

    leaq (AO1, LDA, 4), AO1
    addq $16*SIZE, BO

    ALIGN_4

.L33:
    testq $2, M
    jz .L34

    movups (AO1), %xmm0
    movups (AO1, LDA), %xmm1

    movups %xmm0, 0*SIZE(BO)
    movups %xmm1, 4*SIZE(BO)

    leaq (AO1, LDA, 2), AO1
    addq $8*SIZE, BO

    ALIGN_4

.L34:
    testq $1, M
    jz .L40

    movups (AO1), %xmm0

    movups %xmm0, 0*SIZE(BO)

    #leaq (AO1, LDA, 1), AO1
    addq $4*SIZE, BO

    ALIGN_4

.L40:
    testq $2, N
    jz .L50

    movq    AO,   AO1
    addq    $2*SIZE, AO
    movq    M,    I
    sarq    $3,   I
    testq   I,    I
    jle .L42

    ALIGN_4

.L41:
    movq (AO1), %mm0
    movq (AO1, LDA), %mm1
    movq (AO1, LDA, 2), %mm2
    movq (AO1, LDA3), %mm3
    movq (AO1, LDA, 4), %mm4
    movq (AO1, LDA5), %mm5
    movq (AO1, LDA3, 2), %mm6
    movq (AO1, LDA7), %mm7

    movq %mm0, 0*SIZE(BO)
    movq %mm1, 2*SIZE(BO)
    movq %mm2, 4*SIZE(BO)
    movq %mm3, 6*SIZE(BO)
    movq %mm4, 8*SIZE(BO)
    movq %mm5, 10*SIZE(BO)
    movq %mm6, 12*SIZE(BO)
    movq %mm7, 14*SIZE(BO)

    leaq (AO1, LDA, 8), AO1
    addq $16*SIZE, BO
    decq I
    testq I, I
    jg .L41

    ALIGN_4

.L42:
    testq $4, M
    jz .L43

    movq (AO1), %mm0
    movq (AO1, LDA), %mm1
    movq (AO1, LDA, 2), %mm2
    movq (AO1, LDA3), %mm3

    movq %mm0, 0*SIZE(BO)
    movq %mm1, 2*SIZE(BO)
    movq %mm2, 4*SIZE(BO)
    movq %mm3, 6*SIZE(BO)

    leaq (AO1, LDA, 4), AO1
    addq $8*SIZE, BO

    ALIGN_4

.L43:
    testq $2, M
    jz .L44

    movq (AO1), %mm0
    movq (AO1, LDA), %mm1

    movq %mm0, 0*SIZE(BO)
    movq %mm1, 2*SIZE(BO)

    leaq (AO1, LDA, 2), AO1
    addq $4*SIZE, BO

    ALIGN_4

.L44:
    testq $1, M
    jz .L50

    movq (AO1), %mm0

    movq %mm0, 0*SIZE(BO)

    #leaq (AO1, LDA, 1), AO1
    addq $2*SIZE, BO

    ALIGN_4

.L50:
    testq $1, N
    jz .L99

    movq    AO,   AO1
    #addq    SIZE, AO
    movq    M,    I
    sarq    $3,   I
    testq   I,    I
    jle .L52

    ALIGN_4

.L51:
    movd (AO1), %mm0
    movd (AO1, LDA), %mm1
    movd (AO1, LDA, 2), %mm2
    movd (AO1, LDA3), %mm3
    movd (AO1, LDA, 4), %mm4
    movd (AO1, LDA5), %mm5
    movd (AO1, LDA3, 2), %mm6
    movd (AO1, LDA7), %mm7

    movd %mm0, 0*SIZE(BO)
    movd %mm1, 1*SIZE(BO)
    movd %mm2, 2*SIZE(BO)
    movd %mm3, 3*SIZE(BO)
    movd %mm4, 4*SIZE(BO)
    movd %mm5, 5*SIZE(BO)
    movd %mm6, 6*SIZE(BO)
    movd %mm7, 7*SIZE(BO)

    leaq (AO1, LDA, 8), AO1
    addq $8*SIZE, BO
    decq I
    testq I, I
    jg .L51

    ALIGN_4

.L52:
    testq $4, M
    jz .L53

    movd (AO1), %mm0
    movd (AO1, LDA), %mm1
    movd (AO1, LDA, 2), %mm2
    movd (AO1, LDA3), %mm3

    movd %mm0, 0*SIZE(BO)
    movd %mm1, 1*SIZE(BO)
    movd %mm2, 2*SIZE(BO)
    movd %mm3, 3*SIZE(BO)

    leaq (AO1, LDA, 4), AO1
    addq $4*SIZE, BO

    ALIGN_4

.L53:
    testq $2, M
    jz .L54

    movd (AO1), %mm0
    movd (AO1, LDA), %mm1

    movd %mm0, 0*SIZE(BO)
    movd %mm1, 1*SIZE(BO)

    leaq (AO1, LDA, 2), AO1
    addq $2*SIZE, BO

    ALIGN_4

.L54:
    testq $1, M
    jz .L99

    movd (AO1), %mm0

    movd %mm0, 0*SIZE(BO)

    #leaq (AO1, LDA, 1), AO1
    #addq SIZE, BO

    ALIGN_4

.L99:
#ifdef WINDOWS_ABI
    movups      0(%rsp), %xmm6
    movups     16(%rsp), %xmm7
    movups     32(%rsp), %xmm8
    movups     48(%rsp), %xmm9
    movups     64(%rsp), %xmm10
    movups     80(%rsp), %xmm11
    movups     96(%rsp), %xmm12
    movups    112(%rsp), %xmm13
    movups    128(%rsp), %xmm14
    movups    144(%rsp), %xmm15

    addq    $STACKSIZE, %rsp
#endif

    popq    %rbx
    popq    %rbp
    popq    %r12
    popq    %r13
    popq    %r14
    popq    %r15
#ifdef WINDOWS_ABI
    popq    %rsi
    popq    %rdi
#endif

    ret

    EPILOGUE
