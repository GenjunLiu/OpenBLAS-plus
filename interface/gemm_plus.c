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

#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define ERROR_NAME "GEMM_PLUS "

#ifndef CBLAS

void NAME(char *TRANSA, char *TRANSB,
	  blasint *M, blasint *N, blasint *K,
	  FLOAT *alpha,
	  FLOAT *a, blasint *ldA,
	  FLOAT *b, blasint *ldB,
	  FLOAT *beta,
	  FLOAT *c, blasint *ldC) {
  blas_arg_t args;
  int transa, transb, nrowa, nrowb;
  blasint info;

  char transA, transB;
  //FLOAT *buffer;
  //FLOAT *sa, *sb;

  args.m = *M;
  args.n = *N;
  args.k = *K;

  args.a = (void *)a;
  args.b = (void *)b;
  args.c = (void *)c;

  args.lda = *ldA;
  args.ldb = *ldB;
  args.ldc = *ldC;

  args.alpha = (void *)alpha;
  args.beta  = (void *)beta;

  transA = *TRANSA;
  transB = *TRANSB;

  TOUPPER(transA);
  TOUPPER(transB);

  transa = -1;
  transb = -1;

  if (transA == 'N') transa = 0;
  if (transA == 'T') transa = 1;
  if (transB == 'N') transb = 0;
  if (transB == 'T') transb = 1;

  nrowa = args.m;
  if (transa & 1) nrowa = args.k;
  nrowb = args.k;
  if (transb & 1) nrowb = args.n;

  info = 0;

#ifdef INT16
  // only support A * B ^ T, alpha is fixed-point decimal bits
  if (*alpha < 0 || *alpha>16) info = 17;
  //if (*beta != 1)        info = 16;
#endif
  if (transa & 1)        info = 15;
  if (!(transb & 1))     info = 14;
  if (args.ldc < args.n) info = 13;
  if (args.ldb < args.k) info = 10;
  if (args.lda < args.k) info =  8;
  if (args.k < 0)        info =  5;
  if (args.n < 0)        info =  4;
  if (args.m < 0)        info =  3;
  if (transb < 0)        info =  2;
  if (transa < 0)        info =  1;

  if (info){
    BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME));
    return;
  }

#else

void CNAME(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
	   blasint m, blasint n, blasint k,
	   FLOAT alpha,
	   FLOAT *a, blasint lda,
	   FLOAT *b, blasint ldb,
	   FLOAT beta,
	   FLOAT *c, blasint ldc) {
  blas_arg_t args;
  int transa, transb;
  blasint nrowa, nrowb, info;

  //FLOAT *buffer;
  //FLOAT *sa, *sb;

  args.alpha = (void *)&alpha;
  args.beta  = (void *)&beta;

  transa = -1;
  transb = -1;
  info   =  0;

  args.m = m;
  args.n = n;
  args.k = k;

  args.a = (void *)a;
  args.b = (void *)b;
  args.c = (void *)c;

  args.lda = lda;
  args.ldb = ldb;
  args.ldc = ldc;

  if (TransA == CblasNoTrans)     transa = 0;
  if (TransA == CblasTrans)       transa = 1;
  if (TransB == CblasNoTrans)     transb = 0;
  if (TransB == CblasTrans)       transb = 1;

  nrowa = args.m;
  if (transa & 1) nrowa = args.k;
  nrowb = args.k;
  if (transb & 1) nrowb = args.n;

  info = -1;

#ifdef INT16
  // only support A * B ^ T, alpha is fixed-point decimal bits
  if (order != CblasRowMajor) info = 18;
  if (alpha < 0 || alpha>16)  info = 17;
  //if (*beta != 1)        info = 16;
#endif
  if (transa & 1)        info = 15;
  if (!(transb & 1))     info = 14;
  if (args.ldc < args.n) info = 13;
  if (args.ldb < args.k) info = 10;
  if (args.lda < args.k) info =  8;
  if (args.k < 0)        info =  5;
  if (args.n < 0)        info =  4;
  if (args.m < 0)        info =  3;
  if (transb < 0)        info =  2;
  if (transa < 0)        info =  1;

  if (info >= 0) {
      BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME));
      return;
  }
#endif

  if ((args.m == 0) || (args.n == 0)) return;

#if 0
  fprintf(stderr, "m = %4d  n = %d  k = %d  lda = %4d  ldb = %4d  ldc = %4d\n",
	 args.m, args.n, args.k, args.lda, args.ldb, args.ldc);
#endif

  IDEBUG_START;

  FUNCTION_PROFILE_START();

  //buffer = (FLOAT *)blas_memory_alloc(0);

  //sa = (FLOAT *)((BLASLONG)buffer +GEMM_OFFSET_A);
  //sb = (FLOAT *)(((BLASLONG)sa + ((GEMM_P * GEMM_Q * COMPSIZE * SIZE + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);

  GEMM_NT_PLUS(&args, NULL, NULL, NULL/*sa*/, NULL/*sb*/, 0);

  //blas_memory_free(buffer);

  FUNCTION_PROFILE_END(COMPSIZE * COMPSIZE, args.m * args.k + args.k * args.n + args.m * args.n, 2 * args.m * args.n * args.k);

  IDEBUG_END;

  return;
}
