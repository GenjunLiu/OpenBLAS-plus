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

CTEST(sgemm_plus, sgemm_inc_0)
{
	char transA = 'N', transB = 'T';
	const blasint M = 31, N = 63, K = 255;
	blasint lda = K, ldb = K, ldc = N;
	float alpha = 1.23f, beta = 0.3453f;
	float A[M*K], B[N*K], C[M*N], C2[M*N];
	int i, j, k;
	for (i=0; i<M*K; i++) A[i]  = (i - M*K/2)/(M*K/3);
	for (i=0; i<N*K; i++) B[i]  = (i - N*K/2)/(N*K/3);
	for (i=0; i<M*N; i++) C[i]  = (i - M*N/2)/(M*N/3);
	for (i=0; i<M*N; i++) C2[i] = (i - M*N/2)/(M*N/3);
	for (i=0; i<M; i++) {
		for (j=0; j<N; j++) {
			float cc = 0.0f;
			for (k=0; k<K; k++) cc += A[i*lda + k] * B[j*ldb + k];
			C2[i*ldc + j] = alpha * cc + beta * C2[i*ldc + j];
		}
	}

	//OpenBLAS
	BLASFUNC(sgemm_plus)(&transA, &transB, (blasint*)&M, (blasint*)&N, (blasint*)&K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

	for(i=0; i<M*N; i++){
		ASSERT_DBL_NEAR_TOL(C2[i], C[i], SINGLE_EPS);
	}
}

CTEST(sgemm_plus, sgemm_inc_1)
{
	char transA = 'N', transB = 'T';
	const blasint M = 32, N = 64, K = 256;
	blasint lda = K, ldb = K, ldc = N;
	float alpha = 1.23f, beta = 0.3453f;
	float A[M*K], B[N*K], C[M*N], C2[M*N];
	int i, j, k;
	for (i=0; i<M*K; i++) A[i]  = (i - M*K/2)/(M*K/3);
	for (i=0; i<N*K; i++) B[i]  = (i - N*K/2)/(N*K/3);
	for (i=0; i<M*N; i++) C[i]  = (i - M*N/2)/(M*N/3);
	for (i=0; i<M*N; i++) C2[i] = (i - M*N/2)/(M*N/3);
	for (i=0; i<M; i++) {
		for (j=0; j<N; j++) {
			float cc = 0.0f;
			for (k=0; k<K; k++) cc += A[i*lda + k] * B[j*ldb + k];
			C2[i*ldc + j] = alpha * cc + beta * C2[i*ldc + j];
		}
	}

	//OpenBLAS
	BLASFUNC(sgemm_plus)(&transA, &transB, (blasint*)&M, (blasint*)&N, (blasint*)&K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

	for(i=0; i<M*N; i++){
		ASSERT_DBL_NEAR_TOL(C2[i], C[i], SINGLE_EPS);
	}
}

CTEST(sgemm_plus, sgemm_inc_2)
{
	char transA = 'N', transB = 'T';
	const blasint M = 257, N = 258, K = 256;
	blasint lda = K, ldb = K, ldc = N;
	float alpha = 1.23f, beta = 0.3453f;
	float A[M*K], B[N*K], C[M*N], C2[M*N];
	int i, j, k;
	for (i=0; i<M*K; i++) A[i]  = (i - M*K/2)/(M*K/3);
	for (i=0; i<N*K; i++) B[i]  = (i - N*K/2)/(N*K/3);
	for (i=0; i<M*N; i++) C[i]  = (i - M*N/2)/(M*N/3);
	for (i=0; i<M*N; i++) C2[i] = (i - M*N/2)/(M*N/3);
	for (i=0; i<M; i++) {
		for (j=0; j<N; j++) {
			float cc = 0.0f;
			for (k=0; k<K; k++) cc += A[i*lda + k] * B[j*ldb + k];
			C2[i*ldc + j] = alpha * cc + beta * C2[i*ldc + j];
		}
	}

	//OpenBLAS
	BLASFUNC(sgemm_plus)(&transA, &transB, (blasint*)&M, (blasint*)&N, (blasint*)&K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

	for(i=0; i<M*N; i++){
		ASSERT_DBL_NEAR_TOL(C2[i], C[i], SINGLE_EPS);
	}
}

/*
	CblasRowMajor
	A: M x K
	B: K x N
	C: A * B : M x N
*/
CTEST(sgemm_plus, sgemm_inc_row_major)
{
	float alpha = 1.23f, beta = 0.3453f;
	blasint M = 211, Max_N = 512, K = 178;

	float *AMeta = (float *)malloc(M*K*sizeof(float)), *BMeta = (float *)malloc(Max_N*K*sizeof(float));
	float *CMeta_SGEMM = (float*)malloc(M*Max_N*sizeof(float)), *CMeta_PRE = (float*)malloc(M*Max_N*sizeof(float));

	int i, j, k;
	blasint n;
	for (i=0; i<M*K; i++) 	   AMeta[i] = (i - M*K/2.0f)/(M*K);
	for (i=0; i<Max_N*K; i++)  BMeta[i] = (i - Max_N*K/2.0f)/(Max_N*K);

	{
		CBLAS_TRANSPOSE transA = CblasNoTrans, transB = CblasTrans;

		for (n=4; n<=Max_N; n = (blasint)(n*1.5 + 1)) {
			--n;
			blasint lda = transA == CblasNoTrans ? K : M, ldb = transB == CblasNoTrans ? n : K, ldc = n;
			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					CMeta_PRE[i*ldc + j] = CMeta_SGEMM[i*ldc + j] = (i*j - M*n/2.0f)/(M*n);
				}
			}
			//OpenBLAS baseline
			cblas_sgemm_plus(CblasRowMajor, transA, transB, M, n, K, alpha, AMeta, lda, BMeta, ldb, beta, CMeta_SGEMM, ldc);

			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					float cc = 0.0f;
					if (transA == CblasNoTrans && transB == CblasNoTrans) {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[k*ldb + j];
					} else if (transA == CblasNoTrans && transB == CblasTrans) {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[j*ldb + k];
					} else if (transA == CblasTrans && transB == CblasNoTrans) {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[k*ldb + j];
					} else if (transA == CblasTrans && transB == CblasTrans) {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[j*ldb + k];
					}
					CMeta_PRE[i*ldc + j] = alpha * cc + beta * CMeta_PRE[i*ldc + j];

					ASSERT_DBL_NEAR_TOL(CMeta_PRE[i*ldc + j], CMeta_SGEMM[i*ldc + j], SINGLE_EPS);
				}
			}
		}
	}

	if (AMeta) free(AMeta), AMeta = NULL;
	if (BMeta) free(BMeta), BMeta = NULL;
	if (CMeta_SGEMM) free(CMeta_SGEMM), CMeta_SGEMM = NULL;
	if (CMeta_PRE) free(CMeta_PRE), CMeta_PRE = NULL;
}
