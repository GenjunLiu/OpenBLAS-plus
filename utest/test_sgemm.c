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

#define ALIGN_PTR(PTR, ALIGN, type)  (type*)(((intptr_t)PTR + ALIGN) & (~ALIGN))

/*
	CblasColMajor
	A: K x M
	B: N x K
	C: B * A : N x M
*/
CTEST(sgemm, sgemm_inc_blasfunc)
{
	float alpha = 1.23f, beta = 0.3453f;
	blasint M = 211, Max_N = 512, K = 178;

	float *AMeta = (float *)malloc(M*K*sizeof(float)), *BMeta = (float *)malloc(Max_N*K*sizeof(float));
	float *CMeta_SGEMM = (float*)malloc(M*Max_N*sizeof(float)), *CMeta_PRE = (float*)malloc(M*Max_N*sizeof(float));

	int i, j, k, s;
	blasint n;
	for (i=0; i<M*K; i++) 	   AMeta[i] = (i - M*K/2.0f)/(M*K);
	for (i=0; i<Max_N*K; i++)  BMeta[i] = (i - Max_N*K/2.0f)/(Max_N*K);


	for (s=0; s<4; s++) {
		char transA = (s&1) ? 'T' : 'N', transB = (s&2) ? 'T' : 'N';

		for (n=4; n<=Max_N; n = (blasint)(n*1.5 + 1)) {
			--n;
			blasint lda = transA == 'N' ? M : K, ldb = transB == 'N' ? K : n, ldc = M;
			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					CMeta_PRE[j*ldc + i] = CMeta_SGEMM[j*ldc + i] = (i*j - M*n/2.0f)/(M*n);
				}
			}
			//OpenBLAS baseline
			BLASFUNC(sgemm)(&transA, &transB, (blasint*)&M, (blasint*)&n, (blasint*)&K, &alpha, AMeta, &lda, BMeta, &ldb, &beta, CMeta_SGEMM, &ldc);

			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					float cc = 0.0f;
					if (transA == 'N' && transB == 'N') {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[j*ldb + k];
					} else if (transA == 'N' && transB == 'T') {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[k*ldb + j];
					} else if (transA == 'T' && transB == 'N') {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[j*ldb + k];
					} else if (transA == 'T' && transB == 'T') {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[k*ldb + j];
					}
					CMeta_PRE[j*ldc + i] = alpha * cc + beta * CMeta_PRE[j*ldc + i];

					ASSERT_DBL_NEAR_TOL(CMeta_PRE[j*ldc + i], CMeta_SGEMM[j*ldc + i], SINGLE_EPS);
				}
			}
		}
	}

	if (AMeta) free(AMeta), AMeta = NULL;
	if (BMeta) free(BMeta), BMeta = NULL;
	if (CMeta_SGEMM) free(CMeta_SGEMM), CMeta_SGEMM = NULL;
	if (CMeta_PRE) free(CMeta_PRE), CMeta_PRE = NULL;
}

/*
	CblasColMajor
	A: K x M
	B: N x K
	C: B * A : N x M
*/
CTEST(sgemm, sgemm_inc_col_major)
{
	float alpha = 1.23f, beta = 0.3453f;
	blasint M = 211, Max_N = 512, K = 178;

	float *AMeta = (float *)malloc(M*K*sizeof(float)), *BMeta = (float *)malloc(Max_N*K*sizeof(float));
	float *CMeta_SGEMM = (float*)malloc(M*Max_N*sizeof(float)), *CMeta_PRE = (float*)malloc(M*Max_N*sizeof(float));

	int i, j, k, s;
	blasint n;
	for (i=0; i<M*K; i++) 	   AMeta[i] = (i - M*K/2.0f)/(M*K);
	for (i=0; i<Max_N*K; i++)  BMeta[i] = (i - Max_N*K/2.0f)/(Max_N*K);

	for (s=0; s<4; s++) {
		CBLAS_TRANSPOSE transA = (s&1) ? CblasTrans : CblasNoTrans, transB = (s&2) ? CblasTrans : CblasNoTrans;

		for (n=4; n<=Max_N; n = (blasint)(n*1.5 + 1)) {
			--n;
			blasint lda = transA == CblasNoTrans ? M : K, ldb = transB == CblasNoTrans ? K : n, ldc = M;
			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					CMeta_PRE[j*ldc + i] = CMeta_SGEMM[j*ldc + i] = (i*j - M*n/2.0f)/(M*n);
				}
			}
			//OpenBLAS baseline
			cblas_sgemm(CblasColMajor, transA, transB, M, n, K, alpha, AMeta, lda, BMeta, ldb, beta, CMeta_SGEMM, ldc);

			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					float cc = 0.0f;
					if (transA == CblasNoTrans && transB == CblasNoTrans) {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[j*ldb + k];
					} else if (transA == CblasNoTrans && transB == CblasTrans) {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[k*ldb + j];
					} else if (transA == CblasTrans && transB == CblasNoTrans) {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[j*ldb + k];
					} else if (transA == CblasTrans && transB == CblasTrans) {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[k*ldb + j];
					}
					CMeta_PRE[j*ldc + i] = alpha * cc + beta * CMeta_PRE[j*ldc + i];

					ASSERT_DBL_NEAR_TOL(CMeta_PRE[j*ldc + i], CMeta_SGEMM[j*ldc + i], SINGLE_EPS);
				}
			}
		}
	}

	if (AMeta) free(AMeta), AMeta = NULL;
	if (BMeta) free(BMeta), BMeta = NULL;
	if (CMeta_SGEMM) free(CMeta_SGEMM), CMeta_SGEMM = NULL;
	if (CMeta_PRE) free(CMeta_PRE), CMeta_PRE = NULL;
}

/*
	CblasRowMajor
	A: M x K
	B: K x N
	C: A * B : M x N
*/
CTEST(sgemm, sgemm_inc_row_major)
{
	float alpha = 1.23f, beta = 0.3453f;
	blasint M = 211, Max_N = 512, K = 178;

	float *AMeta = (float *)malloc(M*K*sizeof(float)), *BMeta = (float *)malloc(Max_N*K*sizeof(float));
	float *CMeta_SGEMM = (float*)malloc(M*Max_N*sizeof(float)), *CMeta_PRE = (float*)malloc(M*Max_N*sizeof(float));

	int i, j, k, s;
	blasint n;
	for (i=0; i<M*K; i++) 	   AMeta[i] = (i - M*K/2.0f)/(M*K);
	for (i=0; i<Max_N*K; i++)  BMeta[i] = (i - Max_N*K/2.0f)/(Max_N*K);

	for (s=0; s<4; s++) {
		CBLAS_TRANSPOSE transA = (s&1) ? CblasTrans : CblasNoTrans, transB = (s&2) ? CblasTrans : CblasNoTrans;

		for (n=4; n<=Max_N; n = (blasint)(n*1.5 + 1)) {
			--n;
			blasint lda = transA == CblasNoTrans ? K : M, ldb = transB == CblasNoTrans ? n : K, ldc = n;
			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					CMeta_PRE[i*ldc + j] = CMeta_SGEMM[i*ldc + j] = (i*j - M*n/2.0f)/(M*n);
				}
			}
			//OpenBLAS baseline
			cblas_sgemm(CblasRowMajor, transA, transB, M, n, K, alpha, AMeta, lda, BMeta, ldb, beta, CMeta_SGEMM, ldc);

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

// conflict with precopy
#ifndef SMP_SERVER

/*
	CblasColMajor
	A: K x M
	B: N x K
	C: B * A : N x M
*/
CTEST(sgemm, sgemm_inc_precopy_blasfunc)
{
	float alpha = 1.23f, beta = 0.3453f;
	blasint M = 211, Max_N = 512, K = 178;
    blasint align_param = 0;
    BLASFUNC(sgemm_get_align)(&align_param);
	float *AMeta = (float *)malloc(M*K*sizeof(float)), *A_PRECOPY = (float *)malloc(M*K*sizeof(float) + align_param), *BMeta = (float *)malloc(Max_N*K*sizeof(float));
	float *CMeta_SGEMM = (float*)malloc(M*Max_N*sizeof(float)), *CMeta_PRE = (float*)malloc(M*Max_N*sizeof(float));
	float *A_PRECOPY_ALIGN = ALIGN_PTR(A_PRECOPY, align_param, float);

	int i, j, k, s;
	blasint n;
	for (i=0; i<M*K; i++) 	   AMeta[i] = (i - M*K/2.0f)/(M*K);
	for (i=0; i<Max_N*K; i++)  BMeta[i] = (i - Max_N*K/2.0f)/(Max_N*K);

	for (s=0; s<4; s++) {
		char transA = (s&1) ? 'T' : 'N', transB = (s&2) ? 'T' : 'N';

		blasint lda = transA == 'N' ? M : K;
		BLASFUNC(sgemm_precopy)(&transA, &transB, (blasint*)&M, (blasint*)&K, AMeta, &lda, A_PRECOPY_ALIGN);

		for (n=4; n<=Max_N; n = (blasint)(n*1.5 + 1)) {
			--n;
			blasint lda = transA == 'N' ? M : K, ldb = transB == 'N' ? K : n, ldc = M;
			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					CMeta_PRE[j*ldc + i] = CMeta_SGEMM[j*ldc + i] = (i*j - M*n/2.0f)/(M*n);
				}
			}
			//OpenBLAS baseline
			BLASFUNC(sgemm_mul)(&transA, &transB, (blasint*)&M, (blasint*)&n, (blasint*)&K, &alpha, A_PRECOPY_ALIGN, &lda, BMeta, &ldb, &beta, CMeta_SGEMM, &ldc);

			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					float cc = 0.0f;
					if (transA == 'N' && transB == 'N') {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[j*ldb + k];
					} else if (transA == 'N' && transB == 'T') {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[k*ldb + j];
					} else if (transA == 'T' && transB == 'N') {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[j*ldb + k];
					} else if (transA == 'T' && transB == 'T') {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[k*ldb + j];
					}
					CMeta_PRE[j*ldc + i] = alpha * cc + beta * CMeta_PRE[j*ldc + i];

					ASSERT_DBL_NEAR_TOL(CMeta_PRE[i], CMeta_SGEMM[i], SINGLE_EPS);
				}
			}
		}
	}

	if (AMeta) free(AMeta), AMeta = NULL;
	if (A_PRECOPY) free(A_PRECOPY), A_PRECOPY = NULL;
	if (BMeta) free(BMeta), BMeta = NULL;
	if (CMeta_SGEMM) free(CMeta_SGEMM), CMeta_SGEMM = NULL;
	if (CMeta_PRE) free(CMeta_PRE), CMeta_PRE = NULL;
}

/*
	CblasColMajor
	A: K x M
	B: N x K
	C: B * A : N x M
*/
CTEST(sgemm, sgemm_inc_precopy_col_major)
{
	float alpha = 1.23f, beta = 0.3453f;
	blasint M = 211, Max_N = 512, K = 178;
    blasint align_param = 0;
    cblas_sgemm_get_align(&align_param);
	float *AMeta = (float *)malloc(M*K*sizeof(float)), *A_PRECOPY = (float *)malloc(M*K*sizeof(float) + align_param), *BMeta = (float *)malloc(Max_N*K*sizeof(float));
	float *CMeta_SGEMM = (float*)malloc(M*Max_N*sizeof(float)), *CMeta_PRE = (float*)malloc(M*Max_N*sizeof(float));
	float *A_PRECOPY_ALIGN = ALIGN_PTR(A_PRECOPY, align_param, float);

	int i, j, k, s;
	blasint n;
	for (i=0; i<M*K; i++) 	   AMeta[i] = (i - M*K/2.0f)/(M*K);
	for (i=0; i<Max_N*K; i++)  BMeta[i] = (i - Max_N*K/2.0f)/(Max_N*K);

	for (s=0; s<4; s++) {
		CBLAS_TRANSPOSE transA = (s&1) ? CblasTrans : CblasNoTrans, transB = (s&2) ? CblasTrans : CblasNoTrans;

		blasint lda = transA == CblasNoTrans ? M : K;
		cblas_sgemm_precopy(CblasColMajor, transA, transB, M, K, AMeta, lda, A_PRECOPY_ALIGN);

		for (n=4; n<=Max_N; n = (blasint)(n*1.5 + 1)) {
			--n;
			blasint lda = transA == CblasNoTrans ? M : K, ldb = transB == CblasNoTrans ? K : n, ldc = M;
			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					CMeta_PRE[j*ldc + i] = CMeta_SGEMM[j*ldc + i] = (i*j - M*n/2.0f)/(M*n);
				}
			}
			//OpenBLAS baseline
			cblas_sgemm_mul(CblasColMajor, transA, transB, M, n, K, alpha, A_PRECOPY_ALIGN, lda, BMeta, ldb, beta, CMeta_SGEMM, ldc);

			for (i=0; i<M; i++) {
				for (j=0; j<n; j++) {
					float cc = 0.0f;
					if (transA == CblasNoTrans && transB == CblasNoTrans) {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[j*ldb + k];
					} else if (transA == CblasNoTrans && transB == CblasTrans) {
						for (k=0; k<K; k++) cc += AMeta[k*lda + i] * BMeta[k*ldb + j];
					} else if (transA == CblasTrans && transB == CblasNoTrans) {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[j*ldb + k];
					} else if (transA == CblasTrans && transB == CblasTrans) {
						for (k=0; k<K; k++) cc += AMeta[i*lda + k] * BMeta[k*ldb + j];
					}
					CMeta_PRE[j*ldc + i] = alpha * cc + beta * CMeta_PRE[j*ldc + i];

					ASSERT_DBL_NEAR_TOL(CMeta_PRE[j*ldc + i], CMeta_SGEMM[j*ldc + i], SINGLE_EPS);
				}
			}
		}
	}

	if (AMeta) free(AMeta), AMeta = NULL;
	if (A_PRECOPY) free(A_PRECOPY), A_PRECOPY = NULL;
	if (BMeta) free(BMeta), BMeta = NULL;
	if (CMeta_SGEMM) free(CMeta_SGEMM), CMeta_SGEMM = NULL;
	if (CMeta_PRE) free(CMeta_PRE), CMeta_PRE = NULL;
}

/*
	CblasRowMajor
	A: M x K
	B: K x N
	C: A * B : M x N
*/
CTEST(sgemm, sgemm_inc_precopy_row_major)
{
	float alpha = 1.23f, beta = 0.3453f;
	blasint Max_M = 512, N = 211, K = 178;
    blasint align_param = 0;
    cblas_sgemm_get_align(&align_param);
	float *AMeta = (float *)malloc(Max_M*K*sizeof(float)), *BMeta = (float *)malloc(N*K*sizeof(float)), *B_PRECOPY = (float *)malloc(N*K*sizeof(float) + align_param);
	float *CMeta_SGEMM = (float*)malloc(Max_M*N*sizeof(float)), *CMeta_PRE = (float*)malloc(Max_M*N*sizeof(float));
	float *B_PRECOPY_ALIGN = ALIGN_PTR(B_PRECOPY, align_param, float);

	int i, j, k, s;
	blasint m;
	for (i=0; i<Max_M*K; i++) 	AMeta[i] = (i - Max_M*K/2.0f)/(Max_M*K);
	for (i=0; i<N*K; i++)  		BMeta[i] = (i - N*K/2.0f)/(N*K);

	for (s=0; s<4; s++) {
		CBLAS_TRANSPOSE transA = (s&1) ? CblasTrans : CblasNoTrans, transB = (s&2) ? CblasTrans : CblasNoTrans;

		blasint ldb = transB == CblasNoTrans ? N : K;
		cblas_sgemm_precopy(CblasRowMajor, transA, transB, N, K, BMeta, ldb, B_PRECOPY_ALIGN);

		for (m=4; m<=Max_M; m = (blasint)(m*1.5 + 1)) {
			--m;
			blasint lda = transA == CblasNoTrans ? K : m, ldb = transB == CblasNoTrans ? N : K, ldc = N;
			for (i=0; i<m; i++) {
				for (j=0; j<N; j++) {
					CMeta_PRE[i*ldc + j] = CMeta_SGEMM[i*ldc + j] = (i*j - m*N/2.0f)/(m*N);
				}
			}
			//OpenBLAS baseline
			cblas_sgemm_mul(CblasRowMajor, transA, transB, m, N, K, alpha, AMeta, lda, B_PRECOPY_ALIGN, ldb, beta, CMeta_SGEMM, ldc);

			for (i=0; i<m; i++) {
				for (j=0; j<N; j++) {
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
	if (B_PRECOPY) free(B_PRECOPY), B_PRECOPY = NULL;
	if (BMeta) free(BMeta), BMeta = NULL;
	if (CMeta_SGEMM) free(CMeta_SGEMM), CMeta_SGEMM = NULL;
	if (CMeta_PRE) free(CMeta_PRE), CMeta_PRE = NULL;
}
#endif  // SMP_SERVER

