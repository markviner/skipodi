#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define M_PI 3.14159265358979323846

int N;
int Np;
int K;
double L;
double Lx, Ly, Lz;
double T;
double hx, hy, hz;
double tau;
double a2;
double at;

double u_analytical(double x, double y, double z, double t) {
    double result = sin(M_PI * x / Lx) * sin(M_PI * y / Ly) * 
                    sin(2*M_PI * z / Lz) * cos(at * t + 2*M_PI);
    return result;
}

int main(int argc, char *argv[]) {
    N = 10;
    Np = 1;
    L = 1.0;
    Lx = Ly = Lz = L;
    T = 1.0;
    K = 20;

    int max_num_threads = omp_get_max_threads();
    if (argc > 1) {
        sscanf(argv[1], "%d", &N);
        if (N > 0)
        {
            printf("The size of a mesh: %d^3\n", N);
        } else {
            printf("Error: wrong N: %d\n", N);
            return 1;
        }
    }
    if (argc > 2) {
        sscanf(argv[2], "%f", &L);
        if (L > 0) {
            Lx = Ly = Lz = L;
            printf("Lx = Ly = Lz = %f\n", L);
        } else {
            printf("Error: Wrong L: %f\n", L);
            return 1;
        }
    }
    if (argc > 3) {
        sscanf(argv[3], "%d", &K);
        if (N > 0)
        {
            printf("Number of steps K: %d\n", K);
        } else {
            printf("Error: wrong K: %d\n", K);
            return 1;
        }
    }
    if (argc > 4) {
        sscanf(argv[4], "%d", &Np);
        if (Np < 1 || Np > max_num_threads) {
            printf("Error: incorrect number of threads: %d\n", Np);
            printf("Threads number must be in [1, %d]\n", max_num_threads);
            return 1;
        }
    }

    omp_set_num_threads(Np);

    /* steps */
    hx = Lx / N;
    hy = Ly / N;
    hz = Lz / N;
    tau = T / K;

    a2 = 1 / (M_PI * M_PI);
    at = sqrt(1.0 / (Lx*Lx) + 1.0 / (Ly*Ly) + 1.0 / (Lz*Lz));

/* #pragma omp parallel for */
/* #pragma omp parallel for reduction(max:max) */
/*     for (int i = 0; i < N; ++i) { */
/*         int thread_id = omp_get_thread_num(); */
/*         if (a[i] > max) */
/*             max = a[i]; */
/*         printf("thread %d computes iteration %d, max = %d\n", thread_id, i, max); */
/*     } */

/*     printf("Max: %d\n", max); */
    return 0;
}
