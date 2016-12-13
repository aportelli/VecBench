#!/usr/bin/env bash

CXXFLAGS='-std=c++11 -O3 -W -Wall -march=native -mtune=native -inline-forceinline'
icpc ${CXXFLAGS} benchmark.cpp cmul.cpp -o cmul-icpc 
