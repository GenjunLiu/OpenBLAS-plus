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
#ifdef FUNCTION_PROFILE
#include "functable.h"
#endif

#ifndef COMPLEX
#define SMP_THRESHOLD_MIN 65536.0
#ifdef XDOUBLE
#define ERROR_NAME "QGEMM Pre-Copy "
#elif defined(DOUBLE)
#define ERROR_NAME "DGEMM Pre-Copy "
#else
#define ERROR_NAME "SGEMM Pre-Copy "
#endif
#else
#define SMP_THRESHOLD_MIN 8192.0
#ifndef GEMM3M
#ifdef XDOUBLE
#define ERROR_NAME "XGEMM Pre-Copy "
#elif defined(DOUBLE)
#define ERROR_NAME "ZGEMM Pre-Copy "
#else
#define ERROR_NAME "CGEMM Pre-Copy "
#endif
#else
#ifdef XDOUBLE
#define ERROR_NAME "XGEMM3M Pre-Copy "
#elif defined(DOUBLE)
#define ERROR_NAME "ZGEMM3M Pre-Copy "
#else
#define ERROR_NAME "CGEMM3M Pre-Copy "
#endif
#endif
#endif

#ifndef GEMM_MULTITHREAD_THRESHOLD
#define GEMM_MULTITHREAD_THRESHOLD 4
#endif

static int (*gemm_precopy[])(FLOAT *, BLASLONG, BLASLONG, BLASLONG, BLASLONG *, FLOAT *) = {
  GEMM_NN, GEMM_TN, GEMM_NT, GEMM_TT
};

#ifndef CBLAS

void NAME(char *TRANSA, char *TRANSB,
	  blasint *M, blasint *K,
	  FLOAT *a, blasint *ldA,
	  FLOAT *c){
  blasint m, k, lda;
  int transa, transb, nrowa, info;

  char transA, transB;

  PRINT_DEBUG_NAME;

  m = *M;
  k = *K;
  lda = *ldA;
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

  nrowa = m;
  if (transa & 1) nrowa = k;

  info = 0;

  if ((intptr_t)c & GEMM_PRECOPY_ALIGN) info =  6;
  if (lda < nrowa)    info =  5;
  if (m < 0)          info =  4;
  if (k < 0)          info =  3;
  if (transb < 0)     info =  2;
  if (transa < 0)     info =  1;

  if (info){
    BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME));
    return;
  }

#else

void CNAME(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
	   blasint m, blasint k,
#ifndef COMPLEX
	   FLOAT *a, blasint lda,
	   FLOAT *c) {
#else
	   void *va, blasint lda,
	   void *vc) {
  FLOAT *a = (FLOAT*) va;
  FLOAT *c = (FLOAT*) vc;	   
#endif

  int transa, transb, nrowa, info;
  PRINT_DEBUG_CNAME;

  transa = -1;
  transb = -1;
  info   = -1;

  if (order == CblasColMajor) {
    if (TransA == CblasNoTrans)     transa = 0;
    if (TransA == CblasTrans)       transa = 1;
    if (TransB == CblasNoTrans)     transb = 0;
    if (TransB == CblasTrans)       transb = 1;

    nrowa = m;
    if (transa & 1) nrowa = k;

    if ((intptr_t)c & GEMM_PRECOPY_ALIGN) info =  6;
    if (lda < nrowa)    info =  5;
    if (m < 0)          info =  4;
    if (k < 0)          info =  3;
    if (transb < 0)     info =  2;
    if (transa < 0)     info =  1;
  }

  if (order == CblasRowMajor) {
    if (TransB == CblasNoTrans)     transa = 0;
    if (TransB == CblasTrans)       transa = 1;
    if (TransA == CblasNoTrans)     transb = 0;
    if (TransA == CblasTrans)       transb = 1;

    nrowa = m;
    if (transa & 1) nrowa = k;

    if ((intptr_t)c & GEMM_PRECOPY_ALIGN) info =  6;
    if (lda < nrowa)    info =  5;
    if (m < 0)          info =  4;
    if (k < 0)          info =  3;
    if (transb < 0)     info =  2;
    if (transa < 0)     info =  1;
  }

  if (info >= 0) {
    BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME));
    return;
  }

#endif

  if (m == 0 || k == 0) return;

  IDEBUG_START;

  FUNCTION_PROFILE_START();

  (gemm_precopy[(transb << 1) | transa])(a, m, k, lda, NULL, c);

  FUNCTION_PROFILE_END(COMPSIZE * COMPSIZE, args.m * args.k + args.k * args.n + args.m * args.n, 2 * args.m * args.n * args.k);

  IDEBUG_END;

  return;
}
