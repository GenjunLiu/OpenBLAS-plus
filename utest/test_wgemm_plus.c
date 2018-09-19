/*****************************************************************************
Copyright (c) 2011-2014, The OpenBLAS Project
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
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

#include "cblas.h"
#include "openblas_utest.h"

CTEST(wgemm_plus, wgemm_inc_0)
{
	char transA = 'N', transB = 'T';
	const blasint M = 31, N = 63, K = 255;
	blasint lda = K, ldb = K, ldc = N;
	int16_t alpha = 6, beta = 1;
	int16_t A[M*K], B[N*K], C[M*N], C2[M*N];
	int i, j, k;
	for (i=0; i<M*K; i++) A[i]  = i - M*K/2;
	for (i=0; i<N*K; i++) B[i]  = i - N*K/2;
	for (i=0; i<M*N; i++) C[i]  = i - M*N/2;
	for (i=0; i<M*N; i++) C2[i] = i - M*N/2;
	for (i=0; i<M; i++) {
		for (j=0; j<N; j++) {
			int cc = 0;
			for (k=0; k<K; k++) cc += A[i*lda + k] * B[j*ldb + k];
			cc = cc >> alpha;
			C2[i*ldc + j] += (int16_t)cc;
		}
	}

	//OpenBLAS
	BLASFUNC(wgemm_plus)(&transA, &transB, (blasint*)&M, (blasint*)&N, (blasint*)&K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

	for(i=0; i<M*N; i++){
		ASSERT_EQUAL(C2[i], C[i]);
	}
}

CTEST(wgemm_plus, wgemm_inc_1)
{
	char transA = 'N', transB = 'T';
	const blasint M = 32, N = 64, K = 256;
	blasint lda = K, ldb = K, ldc = N;
	int16_t alpha = 6, beta = 1;
	int16_t A[M*K], B[N*K], C[M*N], C2[M*N];
	int i, j, k;
	for (i=0; i<M*K; i++) A[i]  = i - M*K/2;
	for (i=0; i<N*K; i++) B[i]  = i - N*K/2;
	for (i=0; i<M*N; i++) C[i]  = i - M*N/2;
	for (i=0; i<M*N; i++) C2[i] = i - M*N/2;
	for (i=0; i<M; i++) {
		for (j=0; j<N; j++) {
			int cc = 0;
			for (k=0; k<K; k++) cc += A[i*lda + k] * B[j*ldb + k];
			cc = cc >> alpha;
			C2[i*ldc + j] += (int16_t)cc;
		}
	}

	//OpenBLAS
	BLASFUNC(wgemm_plus)(&transA, &transB, (blasint*)&M, (blasint*)&N, (blasint*)&K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

	for(i=0; i<M*N; i++){
		ASSERT_EQUAL(C2[i], C[i]);
	}
}

CTEST(wgemm_plus, wgemm_inc_2)
{
	char transA = 'N', transB = 'T';
	const blasint M = 257, N = 258, K = 256;
	blasint lda = K, ldb = K, ldc = N;
	int16_t alpha = 6, beta = 1;
	int16_t A[M*K], B[N*K], C[M*N], C2[M*N];
	int i, j, k;
	for (i=0; i<M*K; i++) A[i]  = i - M*K/2;
	for (i=0; i<N*K; i++) B[i]  = i - N*K/2;
	for (i=0; i<M*N; i++) C[i]  = i - M*N/2;
	for (i=0; i<M*N; i++) C2[i] = i - M*N/2;
	for (i=0; i<M; i++) {
		for (j=0; j<N; j++) {
			int cc = 0;
			for (k=0; k<K; k++) cc += A[i*lda + k] * B[j*ldb + k];
			cc = cc >> alpha;
			C2[i*ldc + j] += (int16_t)cc;
		}
	}

	//OpenBLAS
	BLASFUNC(wgemm_plus)(&transA, &transB, (blasint*)&M, (blasint*)&N, (blasint*)&K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

	for(i=0; i<M*N; i++){
		ASSERT_EQUAL(C2[i], C[i]);
	}
}
