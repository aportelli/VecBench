#include <iostream>
#include "benchmark.hpp"

using namespace std;

int main(void)
{
  bench_info();
  bench_mulgen2();
  bench_mulgen3();
  bench_macgen();
  bench_matmulgen();
  bench_mulstd2();
  bench_mulstd3();
  bench_macstd();
  bench_matmulstd();
  bench_mulavx2();
  bench_mulavx3();
  bench_macavx();
  bench_matmulavx();

  return EXIT_SUCCESS;
}

