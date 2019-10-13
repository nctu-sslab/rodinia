/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**
 * **/
/**   Edited by: Sang-Ha  Lee
 * **/
/**				 University of Virginia
 * **/
/**
 * **/
/**   Description:	No longer supports fuzzy c-means clustering;
 * **/
/**					only regular k-means clustering.
 * **/
/**					Simplified for main functionality: regular
 * k-means	**/
/**					clustering.
 * **/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__device__ float euclid_dist_2(float *pt1, float *pt2, int numdims) {
    int i;
    float ans = 0.0;

    for (i = 0; i < numdims; i++)
        ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

    return (ans);
}

__device__ int find_nearest_point(float *pt,                  /* [nfeatures] */
                       int nfeatures, float **pts, /* [npts][nfeatures] */
                       int npts) {
    int index, i;
    float min_dist = FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i = 0; i < npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures); /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return (index);
}


__global__ void kernel(float **feature,int nfeatures, int nclusters, int npoints, float **clusters, int *membership, float **partial_new_centers_len, float ***partial_new_centers, float *delta) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i >= npoints ) {
        return;
    }
    //int tid = omp_get_thread_num();
    int tid = 0;//omp_get_thread_num();
        /* find the index of nestest cluster centers */
        int index = find_nearest_point(feature[i], nfeatures, clusters,
                                   nclusters);
        /* if membership changes, increase delta by 1 */
        if (membership[i] != index)
            (*delta) += 1.0;

        /* assign the membership to object i */
        membership[i] = index;

        /* update new cluster centers : sum of all objects located
               within */
        partial_new_centers_len[tid][index]++;
        for (int j = 0; j < nfeatures; j++)
            partial_new_centers[tid][index][j] += feature[i][j];
}


extern "C" {
//#include "kmeans.h"
#define CUDA_ERROR_CHECK
#include "/home/pschen/sslab/omp_offloading/include/cuda_check.h"

extern double wtime(void);



/*----< kmeans_clustering() >---------------------------------------------*/
float **kmeans_clustering(float **feature, /* in: [npoints][nfeatures] */
                          int nfeatures, int npoints, int nclusters,
                          float threshold, int *membership) /* out: [npoints] */
{

    int i, j, k, n = 0, index, loop = 0;
    int *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float **new_centers;  /* [nclusters][nfeatures] */
    float **clusters;     /* out: [nclusters][nfeatures] */
    float delta;

    //double timing;

    int nthreads;
    int **partial_new_centers_len;
    float ***partial_new_centers;

    //nthreads = omp_get_max_threads();
    nthreads = 1;

    /* allocate space for returning variable clusters[] */
    clusters = (float **)malloc(nclusters * sizeof(float *));
    clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        // n = (int)rand() % npoints;
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int *)calloc(nclusters, sizeof(int));

    new_centers = (float **)malloc(nclusters * sizeof(float *));
    new_centers[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));
    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;


    partial_new_centers_len = (int **)malloc(nthreads * sizeof(int *));
    partial_new_centers_len[0] =
        (int *)calloc(nthreads * nclusters, sizeof(int));
    for (i = 1; i < nthreads; i++)
        partial_new_centers_len[i] = partial_new_centers_len[i - 1] + nclusters;

    partial_new_centers = (float ***)malloc(nthreads * sizeof(float **));
    partial_new_centers[0] =
        (float **)malloc(nthreads * nclusters * sizeof(float *));
    for (i = 1; i < nthreads; i++)
        partial_new_centers[i] = partial_new_centers[i - 1] + nclusters;

    for (i = 0; i < nthreads; i++) {
        for (j = 0; j < nclusters; j++)
            partial_new_centers[i][j] =
                (float *)calloc(nfeatures, sizeof(float));
    }

    do {
        index = 0;// redundant
        delta = 0.0 + index;

        float *deltaptr = &delta;

        // 1D function
#define DEEP_COPY1D(NAME, N, TYPE)\
        TYPE *NAME##_d1;\
        {\
            CudaSafeCall(cudaMalloc(&NAME##_d1, N *sizeof(TYPE)));\
            CudaSafeCall(cudaMemcpy(NAME##_d1, NAME, N*sizeof(TYPE), cudaMemcpyHostToDevice));\
        }

#define DEEP_BACK1D(NAME, N, TYPE)\
        CudaSafeCall(cudaMemcpy(NAME, NAME##_d1, N*sizeof(TYPE), cudaMemcpyDeviceToHost));

#define DEEP_FREE1D(NAME)\
        CudaSafeCall(cudaFree(NAME##_d1));

        // 2D function
#define DEEP_COPY2D(NAME, N, M) \
        float **NAME##_d1, **NAME##_h1;\
        float *NAME##_d2;\
        { \
            CudaSafeCall(cudaMalloc(&NAME##_d1, N *sizeof(float*)));\
            CudaSafeCall(cudaMalloc(&NAME##_d2, N * M *sizeof(float)));\
            NAME##_h1 = (float**)malloc(N*sizeof(float*)); \
            for (int i = 0 ; i < N; i++) {\
                NAME##_h1[i] = NAME##_d2 +i* M ;\
                CudaSafeCall(cudaMemcpy(NAME##_d2 + i*M, NAME[i], M*sizeof(float), cudaMemcpyHostToDevice));\
            }\
            CudaSafeCall(cudaMemcpy(NAME##_d1, NAME##_h1, N*sizeof(float*), cudaMemcpyHostToDevice));\
        }

#define DEEP_BACK2D(NAME, N, M)\
        {\
            for (int i = 0; i < N; i++) {\
                CudaSafeCall(cudaMemcpy(NAME[i], NAME##_d2 + i*M, M * sizeof(float), cudaMemcpyDeviceToHost));\
            }\
        }\

#define DEEP_FREE2D(NAME) \
        {\
            CudaSafeCall(cudaFree(NAME##_d1));\
            CudaSafeCall(cudaFree(NAME##_d2));\
            free(NAME##_h1);\
        }

        // 3D function
#define DEEP_COPY3D(NAME, N, M, K)\
        float ***NAME##_d1, ***NAME##_h1;\
        float *NAME##_d3;\
        float **NAME##_d2;\
        {\
            CudaSafeCall(cudaMalloc(&NAME##_d1, N * sizeof(float**)));\
            CudaSafeCall(cudaMalloc(&NAME##_d2, N * M * sizeof(float*)));\
            CudaSafeCall(cudaMalloc(&NAME##_d3, N * M * K * sizeof(float)));\
            /*Host ptr*/\
            float **NAME##_h2 = (float **)malloc(M * sizeof(float*));\
            NAME##_h1 = (float ***)malloc(N * sizeof(float**));\
            for (int i = 0; i < N; i++) {\
                for (int j = 0; j < M; j++) {\
                    CudaSafeCall(cudaMemcpy(NAME##_d3+(i*M+j)*K, &NAME[i][j], K * sizeof(float),cudaMemcpyHostToDevice));\
                    NAME##_h2[j] = NAME##_d3+i*M+j;\
                }\
                CudaSafeCall(cudaMemcpy(NAME##_d2+i*M, NAME##_h2, M * sizeof(float*),cudaMemcpyHostToDevice));\
                NAME##_h1[i] = NAME##_d2+i*M;\
            }\
            CudaSafeCall(cudaMemcpy(NAME##_d1, NAME##_h1, N * sizeof(float**),cudaMemcpyHostToDevice));\
            free(NAME##_h2);\
        }

#define DEEP_BACK3D(NAME, N, M, K)\
        {\
            for (int i = 0; i < N; i ++) {\
                for (int j = 0; j < M; j++) {\
                    CudaSafeCall(cudaMemcpy(NAME[i][j], NAME##_d3+i*M+j, K * sizeof(float), cudaMemcpyDeviceToHost));\
                }\
            }\
        }\


#define DEEP_FREE3D(NAME)\
        CudaSafeCall(cudaFree(NAME##_d3));\
        CudaSafeCall(cudaFree(NAME##_d2));\
        CudaSafeCall(cudaFree(NAME##_d1));\
        free(NAME##_h1);

        // HtoD
        DEEP_COPY1D(deltaptr, 1, float);
        DEEP_COPY1D(membership, npoints, int);

        DEEP_COPY2D(feature, npoints, nfeatures);
        DEEP_COPY2D(clusters, nclusters, nfeatures);
        DEEP_COPY2D(partial_new_centers_len, nthreads, nclusters);

        DEEP_COPY3D(partial_new_centers, nthreads, nclusters, nfeatures);
/*
        puts("before Kernel");
for ( i = 0 ; i < nclusters; i++) {
            for (j = 0; j < nfeatures; j++) {
                            printf("%f ", clusters[i][j]);
                                    }
                    puts("");
                        }
                        */
        // Kernel
        kernel<<<(npoints+511)/512,512>>>(feature_d1, nfeatures, nclusters, npoints, clusters_d1, membership_d1, partial_new_centers_len_d1, partial_new_centers_d1, deltaptr_d1);
        CudaCheckError();
        cudaDeviceSynchronize();

        // DtoH
        DEEP_BACK1D(deltaptr, 1, float);
        DEEP_BACK1D(membership, npoints, int);

        DEEP_BACK2D(feature, npoints, nfeatures);
        DEEP_BACK2D(clusters, nclusters, nfeatures);
        DEEP_BACK2D(partial_new_centers_len, nthreads, nclusters);

        DEEP_BACK3D(partial_new_centers, nthreads, nclusters, nfeatures);


        DEEP_FREE1D(deltaptr);
        DEEP_FREE1D(membership);

        DEEP_FREE2D(feature);
        DEEP_FREE2D(clusters);
        DEEP_FREE2D(partial_new_centers_len);

        DEEP_FREE3D(partial_new_centers);


        /* let the main thread perform the array reduction */
        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nthreads; j++) {
                new_centers_len[i] += partial_new_centers_len[j][i];
                partial_new_centers_len[j][i] = 0.0;
                for (k = 0; k < nfeatures; k++) {
                    new_centers[i][k] += partial_new_centers[j][i][k];
                    partial_new_centers[j][i][k] = 0.0;
                }
            }
        }

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nfeatures; j++) {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0; /* set back to 0 */
            }
            new_centers_len[i] = 0; /* set back to 0 */
        }
    printf("Loop: %d\n", loop);

    } while (delta > threshold && loop++ < 500);


    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
}
