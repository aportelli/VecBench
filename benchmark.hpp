// This file is part of VecBench. Copyright Antonin Portelli 2016-2017
//
// VecBench is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// VecBench is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with VecBench.  If not, see <http://www.gnu.org/licenses/>.

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
