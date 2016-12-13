#!/usr/bin/env bash

CXXFLAGS='-std=c++11 -O3 -W -Wall -march=native -mtune=native'
g++-5 ${CXXFLAGS} benchmark.cpp cmul.cpp -o cmul-gcc-5
