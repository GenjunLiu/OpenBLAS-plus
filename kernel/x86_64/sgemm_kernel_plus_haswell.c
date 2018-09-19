/***************************************************************************
  Copyright (c) 2014, The OpenBLAS Project
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
 *****************************************************************************/

#include "common.h"

int CNAME(BLASLONG m, BLASLONG n, BLASLONG k, float alpha, float* a, BLASLONG lda, float* b, BLASLONG ldb, float* c, BLASLONG ldc) {
    const int kk = k&(~15), kkk = k - (k&15);
    for (int i=0; i<m; i++) {
        float *aa = &a[i*lda];
        for (int j=0; j<n; j++) {
            float cc = 0;
            float *bb = &b[j*ldb];
            for (int x=0; x<kk; x+=16) {
                float c0 = aa[x + 0] * bb[x + 0];
                float c1 = aa[x + 1] * bb[x + 1];
                float c2 = aa[x + 2] * bb[x + 2];
                float c3 = aa[x + 3] * bb[x + 3];
                float c4 = aa[x + 4] * bb[x + 4];
                float c5 = aa[x + 5] * bb[x + 5];
                float c6 = aa[x + 6] * bb[x + 6];
                float c7 = aa[x + 7] * bb[x + 7];
                float c8 = aa[x + 8] * bb[x + 8];
                float c9 = aa[x + 9] * bb[x + 9];
                float c10 = aa[x + 10] * bb[x + 10];
                float c11 = aa[x + 11] * bb[x + 11];
                float c12 = aa[x + 12] * bb[x + 12];
                float c13 = aa[x + 13] * bb[x + 13];
                float c14 = aa[x + 14] * bb[x + 14];
                float c15 = aa[x + 15] * bb[x + 15];
                cc += (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15);
            }
            for (int x=kkk; x<k; x++) {
                cc += aa[x] * bb[x];
            }
            c[i*ldc + j] += alpha * cc; 
        }
    }
    return 0;
}
