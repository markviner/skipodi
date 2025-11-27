#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define M_PI 3.14159265358979323846

const int BENCHMARK = 1;

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

double *u_prev; // u^{n-1}
double *u_curr; // u^n
double *u_next; // u^{n+1}

double *error_history = NULL;

char *file_name;

double u_analytical(double x, double y, double z, double t) {
    double result = sin(M_PI * x / Lx) * sin(M_PI * y / Ly) *
                    sin(2*M_PI * z / Lz) * cos(at * t + 2*M_PI);
    return result;
}

int get_index(int i, int j, int k) {
    return i * (N+1)*(N+1) + j*(N+1) + k;
}


/* phi(x, y, z) = u(x, y, z, 0) */
double phi(double x, double y, double z) {
    return u_analytical(x, y, z, 0.0);
}

double laplacian(double *u, int i, int j, int k) {
    int idx = get_index(i, j, k);
    int idx_xp = get_index(i+1, j, k);
    int idx_xm = get_index(i-1, j, k);
    int idx_yp = get_index(i, j+1, k);
    int idx_ym = get_index(i, j-1, k);
    int idx_zp = get_index(i, j, k+1);
    int idx_zm = get_index(i, j, k-1);

    double lap_x = (u[idx_xp] - 2.0*u[idx] + u[idx_xm]) / (hx * hx);
    double lap_y = (u[idx_yp] - 2.0*u[idx] + u[idx_ym]) / (hy * hy);
    double lap_z = (u[idx_zp] - 2.0*u[idx] + u[idx_zm]) / (hz * hz);

    return lap_x + lap_y + lap_z;
}

void solve_naive() {
    /* u^0 = phi(x,y,z) */
    for(int i = 1; i < N; ++i) {
        for(int j = 1; j < N; ++j) {
            for(int k = 1; k < N; ++k) {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                u_prev[get_index(i, j, k)] = phi(x, y, z);
            }
        }
    }

    /* boundary conditions for u^0 */
    for(int j = 0; j <= N; ++j) {
        for(int k = 0; k <= N; ++k) {
            u_prev[get_index(0, j, k)] = 0.0;
            u_prev[get_index(N, j, k)] = 0.0;
        }
    }

    for(int i = 0; i <= N; ++i) {
        for(int k = 0; k <= N; ++k) {
            u_prev[get_index(i, 0, k)] = 0.0;
            u_prev[get_index(i, N, k)] = 0.0;
        }
    }

    for(int i = 0; i <= N; ++i) {
        for(int j = 0; j <= N; ++j) {
            u_prev[get_index(i, j, 0)] = u_prev[get_index(i, j, N)];
        }
    }

    /* u^1 = u^0 + a^2 * tau^2 / 2 * laplacian(phi) */
    for(int i = 1; i < N; ++i) {
        for(int j = 1; j < N; ++j) {
            for(int k = 1; k < N; ++k) {
                double lap = laplacian(u_prev, i, j, k);
                int idx = get_index(i, j, k);
                u_curr[idx] = u_prev[idx] + 0.5 * a2 * tau * tau * lap;
            }
        }
    }

    /* boundary conditions for u^1 */
    for(int j = 0; j <= N; ++j) {
        for(int k = 0; k <= N; ++k) {
            u_curr[get_index(0, j, k)] = 0.0;
            u_curr[get_index(N, j, k)] = 0.0;
        }
    }

    for(int i = 0; i <= N; ++i) {
        for(int k = 0; k <= N; ++k) {
            u_curr[get_index(i, 0, k)] = 0.0;
            u_curr[get_index(i, N, k)] = 0.0;
        }
    }

    for(int i = 0; i <= N; ++i) {
        for(int j = 0; j <= N; ++j) {
            u_curr[get_index(i, j, 0)] = u_curr[get_index(i, j, N)];
        }
    }

    for(int n = 1; n < K; ++n) {
        /* u^{n+1} = 2*u^n - u^{n-1} + a^2*tau^2 * laplacian(u^n) */
        for(int i = 1; i < N; ++i) {
            for(int j = 1; j < N; ++j) {
                for(int k = 1; k < N; ++k) {
                    double lap = laplacian(u_curr, i, j, k);
                    int idx = get_index(i, j, k);
                    u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + a2 * tau * tau * lap;
                }
            }
        }

        /* boundary conditions for u^{n+1} */
        for(int j = 0; j <= N; ++j) {
            for(int k = 0; k <= N; ++k) {
                u_next[get_index(0, j, k)] = 0.0;
                u_next[get_index(N, j, k)] = 0.0;
            }
        }

        for(int i = 0; i <= N; ++i) {
            for(int k = 0; k <= N; ++k) {
                u_next[get_index(i, 0, k)] = 0.0;
                u_next[get_index(i, N, k)] = 0.0;
            }
        }

        for(int i = 0; i <= N; ++i) {
            for(int j = 0; j <= N; ++j) {
                u_next[get_index(i, j, 0)] = u_next[get_index(i, j, N)];
            }
        }

        /* swap */
        double *temp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = temp;

        if(n % 10 == 0 || n == K-1) {
            printf("Iter %d/%d\n", n, K);
        }
    }
}

int main(int argc, char *argv[]) {
    N = 10;
    Np = 1;
    L = 1.0;
    Lx = Ly = Lz = L;
    T = 1.0;
    K = 20;

    if (argc > 1) {
        sscanf(argv[1], "%d", &N);
        if (N > 0)
        {
            printf("INPUT Mesh Size (N): %d^3\n", N);
        } else {
            printf("Error: wrong N: %d\n", N);
            return 1;
        }
    }
    if (argc > 2) {
        sscanf(argv[2], "%lf", &L);
        if (L > 0) {
            Lx = Ly = Lz = L;
            printf("INPUT Lx = Ly = Lz = %f\n", L);
        } else {
            printf("Error: Wrong L: %f\n", L);
            return 1;
        }
    }
    if (argc > 3) {
        sscanf(argv[3], "%d", &K);
        if (K > 0)
        {
            printf("INPUT Number Of Steps (K): %d\n", K);
        } else {
            printf("Error: wrong K: %d\n", K);
            return 1;
        }
    }

    /* steps */
    hx = Lx / N;
    hy = Ly / N;
    hz = Lz / N;
    tau = T / K;

    a2 = 1 / (M_PI * M_PI);
    at = sqrt(1.0 / (Lx*Lx) + 1.0 / (Ly*Ly) + 4.0 / (Lz*Lz));

    /* Courant condition check */
    double h_min = hx;
    if (hy < h_min) h_min = hy;
    if (hz < h_min) h_min = hz;

    double a = sqrt(a2);
    double tau_max = h_min / (a * sqrt(3.0));
    double courant = tau / tau_max;

    printf("\n=========== Courant Check ===========\n");
    printf("h_min:       %.6f\n", h_min);
    printf("tau_max:     %.6f (for stability)\n", tau_max);
    printf("tau:         %.6f (current)\n", tau);
    printf("Courant num: %.3f (must be < 1.0)\n", courant);

    if (courant >= 1.0) {
        printf("WARNING: Courant unstable!\n");
        printf("Minimum K for stability: %d\n", (int)(T / tau_max) + 1);
    } else {
        printf("Stability: OK\n\n");
    }

    printf("=========== Params ===========\n");
    printf("Mesh:        %d^3 = %d nodes\n", (N+1), (N+1)*(N+1)*(N+1));
    printf("Mesh ratios: %.6f x %.6f x %.6f\n", Lx, Ly, Lz);
    printf("K:           %d\n", K);
    printf("Np:          %d\n", Np);
    printf("Steps:       hx = %.6f, hy = %.6f, hz = %.6f\n", hx, hy, hz);
    printf("params:      at = %.6f, a^2 = %.6f\n", at, a2);

    /* allocate mem */
    int size = (N+1)*(N+1)*(N+1);
    u_prev = (double*)malloc(size * sizeof(double));
    u_curr = (double*)malloc(size * sizeof(double));
    u_next = (double*)malloc(size * sizeof(double));

    if(!u_prev || !u_curr || !u_next) {
        printf("Error: mem allocation fault\n");
        return 1;
    }

    for(int i = 0; i < size; i++) {
        u_prev[i] = 0.0;
        u_curr[i] = 0.0;
        u_next[i] = 0.0;
    }

    clock_t start_time, end_time;
    double time_elapsed;

    printf("Straight solution ... \n");
    printf("+++++ Timer start +++++ \n");
    start_time = clock();
    solve_naive();
    end_time = clock();
    printf("+++++ Time end +++++ \n");
    time_elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Solution time: %.6f\n", time_elapsed);

    free(u_prev);
    free(u_curr);
    free(u_next);
    return 0;
}
