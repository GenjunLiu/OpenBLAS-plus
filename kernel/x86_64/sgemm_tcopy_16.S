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
#define B    ARG5    /* r8  */

#define I    %r10
#define J    %rbp
#define LDA2 %r13

#define AO    %r9
#define AO1   %r11
#define BO    %rax

#define X1    %rbx
#define X2    %r8

#define X1D   %ebx
#define X2D   %r8d

#else

#define STACKSIZE 256

#define M    ARG1    /* rcx */
#define N    ARG2    /* rdx */
#define A    ARG3    /* r8  */
#define LDA  ARG4    /* r9  */
#define OLD_B        40 + 64 + STACKSIZE(%rsp) /* 64 = push 8*8, 40 = return address(8) + ARG1-4(4*8) */

#define B    %rdi

#define I    %r10
#define J    %r11
#define LDA2 %r15

#define AO    %r12
#define AO1   %r13
#define BO    %rax

#define X1    %rbx
#define X2    %rsi

#define X1D   %ebx
#define X2D   %esi

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

    movq    A,  AO
    movq    B,  BO
    movq    LDA,  LDA2
    salq    $1, LDA2
    imulq   $1 * SIZE, LDA2

    movq    N,    J
    sarq    $4,   J
    testq   J,    J
    jle .L20

.L10:
    movq    AO,   AO1
    addq    $16*SIZE, AO
    movq    M,    I
    sarq    $1,   I
    testq   I,    I
    jle .L12

    ALIGN_4

.L11:

    vmovups 0*SIZE(AO1), %ymm0
    vmovups 8*SIZE(AO1), %ymm1
    vmovups 0*SIZE(AO1, LDA, SIZE), %ymm2
    vmovups 8*SIZE(AO1, LDA, SIZE), %ymm3

    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)
    vmovups %ymm2, 16*SIZE(BO)
    vmovups %ymm3, 24*SIZE(BO)

    addq LDA2, AO1
    addq $32*SIZE, BO
    decq I
    testq I, I
    jg .L11

    ALIGN_4

.L12:

    testq $1, M
    jz .L13

    vmovups 0*SIZE(AO1),%ymm0
    vmovups 8*SIZE(AO1),%ymm1
    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)
    addq $16*SIZE, BO

    ALIGN_4
.L13:
    decq J
    testq J, J
    jg .L10

.L20:

    testq $8, N
    jz .L30

    movq    AO,   AO1
    addq    $8*SIZE, AO
    movq    M,    I
    sarq    $1,   I
    testq   I,    I
    jle .L22

    ALIGN_4
.L21:

    vmovups 0*SIZE(AO1), %ymm0
    vmovups 0*SIZE(AO1, LDA, SIZE), %ymm1
    vmovups %ymm0, 0*SIZE(BO)
    vmovups %ymm1, 8*SIZE(BO)

    addq LDA2, AO1
    addq $16*SIZE, BO
    decq I
    testq I, I
    jg .L21

    ALIGN_4
.L22:

    testq $1, M
    jz .L30

    vmovups 0*SIZE(AO1), %ymm0
    vmovups %ymm0, 0*SIZE(BO)

    addq $8*SIZE, BO

    ALIGN_4
.L30:

    testq $4, N
    jz .L40

    movq    AO,   AO1
    addq    $4*SIZE, AO
    movq    M,    I
    sarq    $1,   I
    testq   I,    I
    jle .L32

    ALIGN_4
.L31:

    movups 0*SIZE(AO1), %xmm0
    movups 0*SIZE(AO1, LDA, SIZE), %xmm1
    movups %xmm0, 0*SIZE(BO)
    movups %xmm1, 4*SIZE(BO)

    addq LDA2, AO1
    addq $8*SIZE, BO
    decq I
    testq I, I
    jg .L31

    ALIGN_4
.L32:

    testq $1, M
    jz .L40

    movups 0*SIZE(AO1), %xmm0
    movups %xmm0, 0*SIZE(BO)

    addq $4*SIZE, BO

    ALIGN_4
.L40:

    testq $2, N
    jz .L50

    movq    AO,   AO1
    addq    $2*SIZE, AO
    movq    M,    I
    sarq    $1,   I
    testq   I,    I
    jle .L42

    ALIGN_4
.L41:

    movq 0*SIZE(AO1), X1
    movq 0*SIZE(AO1, LDA, SIZE), X2
    movq X1, 0*SIZE(BO)
    movq X2, 2*SIZE(BO)

    addq LDA2, AO1
    addq $4*SIZE, BO
    decq I
    testq I, I
    jg .L41

    ALIGN_4
.L42:

    testq $1, M
    jz .L50

    movq 0*SIZE(AO1), X1
    movq X1, 0*SIZE(BO)

    addq $2*SIZE, BO

    ALIGN_4
.L50:

    testq $1, N
    jz .L99

    movq    AO,   AO1
    movq    M,    I
    sarq    $1,   I
    testq   I,    I
    jle .L52

.L51:

    movl 0*SIZE(AO1), X1D
    movl 0*SIZE(AO1, LDA, SIZE), X2D
    movl X1D, 0*SIZE(BO)
    movl X2D, 1*SIZE(BO)

    addq LDA2, AO1
    addq $2*SIZE, BO
    decq I
    testq I, I
    jg .L51

    ALIGN_4
.L52:

    testq $1, M
    jz .L99

    movl 0*SIZE(AO1), X1D
    movl X1D, 0*SIZE(BO)

    #addq $1*SIZE, BO

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
