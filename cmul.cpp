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

