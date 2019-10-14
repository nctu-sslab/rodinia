
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
