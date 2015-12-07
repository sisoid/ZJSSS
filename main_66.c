#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <omp.h>

#define COUNTERS 66
#define C_SIZE 64
#define C_STOP 65 // == C_SIZE+1
#define N 4224 // == COUNTERS*C_SIZE
#define N2 17842176 // == N*N

int M[N2];

int uniq(int i, int* counters) {
    for (int j = 1; j <= i - 1; j++) {
        int a = (j - 1) * C_SIZE + counters[j-1];
        int b = (i - 1) * C_SIZE + counters[i-1];
        if (M[(a - 1) + N * (b - 1)] == 0)
            return 1;
    }
    return 0;
}

int searcher(int part_number, FILE *f) {
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
    counters[10] = (part_number - 1) / 64 + 1;
    counters[11] = (part_number - 1) % 64 + 1;

    double iter = 0;
    int current = 1;
    while (1) {
        iter++;

        // stop if search in the selected part is finished
        if (counters[10] != (part_number - 1) / 64 + 1 || counters[11] != (part_number - 1) % 64 + 1) {
            fprintf(f, "Part number %d is done!\nWe have found nothing.", part_number);
            fprintf(f, "\nNumber of iterations: %f\n\n", iter);
            fflush(f);
            break;
        }

        // first subspace is always good
        if (current == 1)
            current = 2;

        // print intermediate state
        if (current == 13 && iter > 1000) {
            fprintf(f, "Current state of part number %d:", part_number);
            for (int i = 0; i < COUNTERS; i++)
                fprintf(f, " %d", counters[i]);
            fprintf(f, "\nNumber of iterations: %f\n\n", iter);
            fflush(f);
        }

        for (int i = current; i <= COUNTERS; i++) {
            if (uniq(i, counters) == 1) {
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

        if (current == COUNTERS && uniq(current, counters) == 0) {
            fprintf(f, "Congratulations! Your code:");
            for (int i = 0; i < COUNTERS; i++)
                fprintf(f, " %d", counters[i]);
            fprintf(f, "\nNumber of iterations: %f\n", iter);
            fflush(f);
            break;
        }
    }

    return 0;
}

int main() {

    // read intersection matrix from file
    FILE *f = fopen("input.tsv", "r");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fscanf(f, "%d", &M[i + N * j]);
        }
    }
    fclose(f);

    f = fopen("output.tsv", "w");

    omp_set_num_threads(24);
    #pragma omp parallel for
    for (int i=0; i<4096; i++) {
        searcher(i+1, f);
    }

    fclose(f);
    return 0;
}
