#!/bin/bash

gcc -fopenmp -O3 main.c -lm && ./a.out 127 1.0 71 8
