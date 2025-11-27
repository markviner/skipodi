#!/bin/bash

echo "n128 naive 1.0"

gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 1
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 2
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 4
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 8
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 1.0 71 16

echo "n256 naive 1.0"

gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 2
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 4
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 8
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 16
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255  1.0 141 32

echo "n128 naive pi"

gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 3.14159265359 71 1
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 3.14159265359 71 2
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 3.14159265359 71 4
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 3.14159265359 71 8
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 127 3.14159265359 71 16

echo "n256 naive pi"

gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255 3.14159265359 141 2
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255 3.14159265359 141 4
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255 3.14159265359 141 8
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255 3.14159265359 141 16
gcc -fopenmp -std=c99 -O3 naive.c -lm && ./a.out 255 3.14159265359 141 32
