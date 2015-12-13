#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <iostream>

#define COUNTERS 66
#define C_SIZE 64
#define C_STOP 65 // == C_SIZE+1
#define N 4224 // == COUNTERS*C_SIZE
#define N2 17842176 // == N*N

#define CUDA_ERROR_CHECK

#define cudaSafeCall(error) __cudaSafeCall(error, __FILE__, __LINE__)
#define cudaCheckErrors() __cudaCheckErrors(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError error, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
    if (error != cudaSuccess) {
        std::cout << "error: CudaSafeCall() failed at " << file << ":" << line
                  << " with \"" << cudaGetErrorString(error) << "\""
                  << std::endl;
        exit(-1);
    }
#endif
}

inline void __cudaCheckErrors(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
    __cudaSafeCall(cudaGetLastError(), file, line);
#endif
}



__device__ inline int uniq(const int* M, int i, int* counters) {
    for (int j = 1; j <= i - 1; j++) {
        int a = (j - 1) * C_SIZE + counters[j-1];
        int b = (i - 1) * C_SIZE + counters[i-1];
        if (M[(a - 1) + N * (b - 1)] == 0)
            return 1;
    }
    return 0;
}

__global__ void searcher(const int* M, int* res, size_t* itersNum) {
    int partNumber = threadIdx.x + blockIdx.x * blockDim.x;
    // initialize counters vector
    int counters[COUNTERS];
    for (int i = 0; i < COUNTERS; i++)
        counters[i] = 1;

    // go to selected part
    counters[0] = 25;
    counters[1] = 5;
    counters[2] = 1;
    counters[3] = 3;
    counters[4] = 4;
    counters[5] = 7;
    counters[6] = 9;
    counters[7] = 2;
    counters[8] = 10;
    counters[9] = 8;
    counters[10] = (partNumber - 1) / 64 + 1;
    counters[11] = (partNumber - 1) % 64 + 1;

    size_t iter = 0;
    size_t current = 1;
    while (1) {
        iter++;

        // stop if search in the selected part is finished
        if (counters[10] != (partNumber - 1) / 64 + 1 || counters[11] != (partNumber - 1) % 64 + 1) {
            for (int i = 0; i < COUNTERS; i++)
                res[partNumber * COUNTERS + i] = -1;
            itersNum[partNumber] = iter;
            break;
        }

        // first subspace is always good
        if (current == 1)
            current = 2;

        // print intermediate state
        // if (current == 13 && iter > 1000) {
        //     fprintf(f, "Current state of part number %d:", partNumber);
        //     for (int i = 0; i < COUNTERS; i++)
        //         fprintf(f, " %d", counters[i]);
        //     fprintf(f, "\nNumber of iterations: %f\n\n", iter);
        //     fflush(f);
        // }

        for (int i = current; i <= COUNTERS; i++) {
            if (uniq(M, i, counters) == 1) {
                counters[i-1]++;
                current = i;
                while (counters[current-1] == C_STOP) {
                    counters[current - 1] = 1;
                    counters[current - 2] = counters[current - 2] + 1;
                    current--;
                }
                break;
            }
        }

        if (current == COUNTERS && uniq(M, current, counters) == 0) {
            for (int i = 0; i < COUNTERS; i++)
                res[partNumber * COUNTERS + i] = counters[i];
            itersNum[partNumber] = iter;
            break;
        }
    }
}

int main() {
    int *M = new int[N2];
    // read intersection matrix from file
    FILE *f = fopen("input.tsv", "r");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fscanf(f, "%d", &M[i + N * j]);
        }
    }
    fclose(f);

    int *d_M;
    cudaMalloc((void **)&d_M, N2 * sizeof(int));
    cudaMemcpy(d_M, M, N2 * sizeof(int), cudaMemcpyHostToDevice);


    int *res = new int[COUNTERS * 4096];
    int *d_res;
    cudaMalloc((void **)&d_res, COUNTERS * 4096 * sizeof(int));

    size_t *itersNum = new size_t[4096];
    memset(itersNum, 0, 4096 * sizeof(size_t));
    size_t *d_itersNum;
    cudaMalloc((void **)&d_itersNum, 4096 * sizeof(size_t));
    cudaMemcpy(d_itersNum, itersNum, 4096 * sizeof(size_t), cudaMemcpyHostToDevice);

    // omp_set_num_threads(24);
    // #pragma omp parallel for
    // for (int i=0; i<4096; i++) {
    //     searcher(d_M, d_res, d_itersNum);
    // }

    searcher<<<32, 128>>>(d_M, d_res, d_itersNum);
    cudaDeviceSynchronize();
    cudaCheckErrors();

    cudaMemcpy(res, d_res, COUNTERS * 4096 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(itersNum, d_itersNum, 4096 * sizeof(size_t), cudaMemcpyDeviceToHost);


    f = fopen("output.tsv", "w");
    for (int partNumber = 0; partNumber < 4096; partNumber++) {
        for (int i = 0; i < COUNTERS; i++) {
            fprintf(f, "%d\t", res[partNumber * COUNTERS + i]);
        }
        fprintf(f, "\n");
        printf("%zu ", itersNum[partNumber]);
    }
    fclose(f);

    cudaFree(d_M);
    cudaFree(d_res);
    cudaFree(d_itersNum);
    delete [] M;
    delete [] res;
    delete [] itersNum;
    return 0;
}
