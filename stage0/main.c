#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int N = 10;
    int input_num_threads = 1;

    int max_num_threads = omp_get_max_threads();
    if (argc > 1) {
        sscanf(argv[1], "%d", &N);
        if (N > 0)
            printf("The size of a mesh: %d^3\n", N);
        else {
            printf("Error: wrong N: %d\n");
            return 1;
        }
    }

    if (argc > 2) {
        sscanf(argv[2], "%d", &input_num_threads);
        if (input_num_threads < 1 || input_num_threads > max_num_threads) {
            printf("Error: incorrect number of threads: %d\n", input_num_threads);
            printf("Threads number must be in [1, %d]\n", max_num_threads);
            return 1;
        }
    }
    /* if theres no 2nd arg -- use deafults */
    omp_set_num_threads(input_num_threads);

#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int thread_id = omp_get_thread_num();
        printf("thread %d computes iteration %d\n", thread_id, i);
    }
    return 0;
}
