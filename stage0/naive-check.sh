#!/bin/bash

echo "n128 naive"

gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 1
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 2
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 4
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 8
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 16

echo "n256 naive"

gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 2
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 4
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 8
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 16
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 32
