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

/* This file is a template for level 3 operation */

#ifndef BETA_OPERATION
#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC) \
  GEMM_BETA((M_TO) - (M_FROM), (N_TO - N_FROM), 0, \
      BETA[0], NULL, 0, NULL, 0, \
      (FLOAT *)(C) + ((M_FROM) + (N_FROM) * (LDC)) * COMPSIZE, LDC)
#endif

#define KERNEL_OPERATION GEMM_KERNEL_NT_PLUS

#ifndef A
#define A	args -> a
#endif
#ifndef LDA
#define LDA	args -> lda
#endif
#ifndef B
#define B	args -> b
#endif
#ifndef LDB
#define LDB	args -> ldb
#endif
#ifndef C
#define C	args -> c
#endif
#ifndef LDC
#define LDC	args -> ldc
#endif
#ifndef M
#define M	args -> m
#endif
#ifndef N
#define N	args -> n
#endif
#ifndef K
#define K	args -> k
#endif

#ifdef TIMING
#define START_RPCC()		rpcc_counter = rpcc()
#define STOP_RPCC(COUNTER)	COUNTER  += rpcc() - rpcc_counter
#else
#define START_RPCC()
#define STOP_RPCC(COUNTER)
#endif

int CNAME(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
		  FLOAT *sa, FLOAT *sb, BLASLONG dummy){
  BLASLONG k, lda, ldb, ldc;
  FLOAT *alpha, *beta;
  FLOAT *a, *b, *c;
  BLASLONG m_from, m_to, n_from, n_to;

#ifdef TIMING
  unsigned long long rpcc_counter;
  unsigned long long innercost  = 0;
  unsigned long long outercost  = 0;
  unsigned long long kernelcost = 0;
  double total;
#endif

  k = K;

  a = (FLOAT *)A;
  b = (FLOAT *)B;
  c = (FLOAT *)C;

  lda = LDA;
  ldb = LDB;
  ldc = LDC;

  alpha = (FLOAT *)args -> alpha;
  beta  = (FLOAT *)args -> beta;

  m_from = 0;
  m_to   = M;

  if (range_m) {
    m_from = *(((BLASLONG *)range_m) + 0);
    m_to   = *(((BLASLONG *)range_m) + 1);
  }

  n_from = 0;
  n_to   = N;

  if (range_n) {
    n_from = *(((BLASLONG *)range_n) + 0);
    n_to   = *(((BLASLONG *)range_n) + 1);
  }

#ifndef INT16
  if (beta && beta[0] != ONE) {
    // CblasRowMajor m is n, n is m
    BETA_OPERATION(n_from, n_to, m_from, m_to, beta, c, ldc);
  }
#endif

  if (alpha[0] == ZERO) return 0;

  BLASLONG l1size = GEMM_PLUS_L1_CACHE_SIZE;
  BLASLONG l2size = GEMM_P * GEMM_Q;
  BLASLONG step_size   = l2size / k / sizeof(FLOAT) / 2;
  BLASLONG b_step_size = l1size / k / sizeof(FLOAT) / 2;
  BLASLONG a_step_size = step_size * 2 - b_step_size;
  if (a_step_size == 0) a_step_size = 1;
  if (b_step_size == 0) b_step_size = 1;

#if 0
  fprintf(stderr, "GEMM(Single): l2size : %ld  step : %ld  k : %ld a_step_size:%d b_step_size:%d \n", l2size, step_size, k, a_step_size, b_step_size);
  fprintf(stderr, "GEMM(Single): M_from : %ld  M_to : %ld  N_from : %ld  N_to : %ld  k : %ld\n", m_from, m_to, n_from, n_to, k);
  fprintf(stderr, "GEMM(Single):: P = %4ld  Q = %4ld  R = %4ld\n", (BLASLONG)GEMM_P, (BLASLONG)GEMM_Q, (BLASLONG)GEMM_R);
	//  fprintf(stderr, "GEMM: SA .. %p  SB .. %p\n", sa, sb);

	//  fprintf(stderr, "A = %p  B = %p  C = %p\n\tlda = %ld  ldb = %ld ldc = %ld\n", a, b, c, lda, ldb, ldc);
#endif

#ifdef TIMING
  innercost = 0;
  outercost = 0;
  kernelcost = 0;
#endif

  BLASLONG i, j;
  BLASLONG min_i, min_j;
  for(i = m_from; i < m_to; i += a_step_size){
    min_i = a_step_size;
    if (i + min_i > m_to) min_i = m_to - i;
    FLOAT *aa = &a[i*lda];
    for(j = n_from; j < n_to; j += b_step_size){
      min_j = b_step_size;
      if (j + min_j > n_to) min_j = n_to - j;
      START_RPCC();
      //fprintf(stderr, "GEMM(Single): min_i:%ld min_j:%ld k:%ld alpha:%ld a:%lld lda:%ld b:%lld ldb:%ld c:%lld ldc:%ld\n",
      //                min_i, min_j, k, alpha[0], (int64_t)aa, lda, (int64_t)&b[j*ldb], ldb, (int64_t)&c[i*ldc + j], ldc);
      KERNEL_OPERATION(min_i, min_j, k, alpha[0], aa, lda, &b[j*ldb], ldb, &c[i*ldc + j], ldc);
      STOP_RPCC(kernelcost);
    }
  }

#ifdef TIMING
  total = (double)outercost + (double)innercost + (double)kernelcost;

  printf( "Copy A : %5.2f Copy  B: %5.2f  Kernel : %5.2f  kernel Effi. : %5.2f Total Effi. : %5.2f\n",
      innercost / total * 100., outercost / total * 100.,
      kernelcost / total * 100.,
      (double)(m_to - m_from) * (double)(n_to - n_from) * (double)k / (double)kernelcost * 100. * (double)COMPSIZE / 2.,
      (double)(m_to - m_from) * (double)(n_to - n_from) * (double)k / total * 100. * (double)COMPSIZE / 2.);

#endif

  return 0;
}
