#ifndef CMUL_BENCHMARKS_HPP_
#define CMUL_BENCHMARKS_HPP_

#include <functional>

// benchmark parameters
constexpr unsigned int nElem = 10000, nIt = 5000, nRow = 3, nCoef = nRow*nRow;

#define DECL_BENCHMARK(name) void bench_##name(void);

void bench_info(void);
DECL_BENCHMARK(addgen);
DECL_BENCHMARK(mulgen2);
DECL_BENCHMARK(mulgen3);
DECL_BENCHMARK(mulstd2);
DECL_BENCHMARK(mulstd3);
DECL_BENCHMARK(mulavx2);
DECL_BENCHMARK(mulavx3);
DECL_BENCHMARK(macgen);
DECL_BENCHMARK(macstd);
DECL_BENCHMARK(macavx);
DECL_BENCHMARK(matmulgen);
DECL_BENCHMARK(matmulstd);
DECL_BENCHMARK(matmulavx);
DECL_BENCHMARK(matmulunrollgen);
DECL_BENCHMARK(matmulunrollavx);

#endif
