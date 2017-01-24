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

#include "benchmark.hpp"
#include <chrono>
#include <functional>
#include <random>
#include <type_traits>
#include "genvec.hpp"
#include "avxvec.hpp"

using namespace std;

// template loop unrollers
template <unsigned int n>
class loop0
{
public:
  template <typename F>
  strong_inline static void exec(F func)
  {
    loop0<n-1>::exec(func);
    func(n - 1);
  }
};

template <>
class loop0<0>
{
public:
  template <typename F>
  strong_inline static void exec(F func  __unused) {}
};

template <unsigned int n>
class loop1
{
public:
  template <typename F>
  strong_inline static void exec(F func)
  {
    loop0<n-1>::exec(func);
    func(n - 1);
  }
};

template <>
class loop1<1>
{
public:
  template <typename F>
  strong_inline static void exec(F func  __unused) {}
};

// type aliases for timers
typedef chrono::high_resolution_clock hrClock;
typedef chrono::time_point<hrClock>   timeStamp;
typedef chrono::duration<double>      duration;

// stack buffers
static vecf    a[nElem*nCoef]   __used,
               b[nElem*nCoef]   __used,  
               c[nElem*nCoef]   __used;
static vecfc  ac[nElem*nCoef]   __used, 
              bc[nElem*nCoef]   __used, 
              cc[nElem*nCoef]   __used;
static __m256 aavx[nElem*nCoef] __used,
              bavx[nElem*nCoef] __used,
              cavx[nElem*nCoef] __used;

// benchmark macro
#define BENCHMARK(name, expr, flops, title)\
void bench_##name(void)\
{\
  timeStamp    start, end;\
  duration     elapsed;\
  \
  cout << "# " << title << endl;\
  start = hrClock::now();\
  for (unsigned int it = 0; it < nIt; ++it)\
  {\
    for (unsigned int i = 0; i < nElem; ++i)\
    {\
      expr\
    }\
  }\
  end     = hrClock::now();\
  elapsed = end - start;\
  cout << "duration= " << elapsed.count() << " s -- ";\
  cout << "Gflop/s= " << (flops)*nIt*nElem/1.0e9/elapsed.count() << endl;\
}

// info message
void bench_info(void)
{
  cout << "##################################" << endl;
  cout << "# VECTORIZATION BENCHMARK         " << endl;
  cout << "# --------------------------------" << endl;
  cout << "# everything is single precision" << endl;
  cout << "# lowercase  : complex numbers" << endl;
  cout << "# uppercase  : " << nRow << "x" << nRow 
                            << " complex matrices" << endl;
  cout << "# generic    : generic SIMD complex multiplication" << endl;
  cout << "# std        : C++ standard library complex multiplication" << endl;
  cout << "# AVX        : AVX+FMA intrinsics complex multiplication" << endl;
  cout << "# array size : " << nElem*nCoef*sizeof(vecf)/1024./1024. 
                            << " Mbytes" << endl;
  cout << "##################################" << endl;
}

// matrix multiplication
#define MATMUL(a, b, c, mul, mac, os)\
for (unsigned int r = 0; r < nRow; ++r)\
for (unsigned int s = 0; s < nRow; ++s)\
{\
  mul(a[os + r*nRow + s], b[os + r*nRow], c[os + s]);\
}\
for (unsigned int r = 0; r < nRow; ++r)\
for (unsigned int t = 1; t < nRow; ++t)\
for (unsigned int s = 0; s < nRow; ++s)\
{\
  mac(a[os + r*nRow + s], b[os + r*nRow + t], c[os + t*nRow + s]);\
}

#define MATMULUNROLL(a, b, c, mul, mac, os)\
loop0<nRow>::exec([os](const unsigned int r)\
{\
  loop0<nRow>::exec([os, r](const unsigned int s)\
  {\
    a[os + r*nRow + s] = mul(b[os + r*nRow], c[os + s]);\
  });\
});\
loop0<nRow>::exec([os](const unsigned int r)\
{\
  loop1<nRow>::exec([os, r](const unsigned int t)\
  {\
    loop0<nRow>::exec([os, r, t](const unsigned int s)\
    {\
      mac(a[os + r*nRow + s], b[os + r*nRow + t], c[os + t*nRow + s]);\
    });\
  });\
});

// benchmarks
BENCHMARK(addgen,
	a[i] = add(b[i], c[i]);
, W<float>::r, "generic a[i] = b[i] + c[i]");

BENCHMARK(mulgen2,
	a[i] = mulgen(b[i], c[i]);
, 6.*W<float>::c, "generic a[i] = b[i]*c[i] (2 args)");

BENCHMARK(mulgen3,
  mulgen(a[i], b[i], c[i]);
, 6.*W<float>::c, "generic a[i] = b[i]*c[i] (3 args)");

BENCHMARK(mulstd2,
	ac[i] = mulstd(bc[i], cc[i]);
, 6.*W<float>::c, "std a[i] = b[i]*c[i] (2 args)");

BENCHMARK(mulstd3,
  mulstd(ac[i], bc[i], cc[i]);
, 6.*W<float>::c, "std a[i] = b[i]*c[i] (3 args)");

BENCHMARK(mulavx2,
  aavx[i] = mulavxf(bavx[i], cavx[i]);
, 6.*W<float>::c, "AVX a[i] = b[i]*c[i] (2 args)");

BENCHMARK(mulavx3,
  aavx[i] = mulavxf(bavx[i], cavx[i]);
, 6.*W<float>::c, "AVX a[i] = b[i]*c[i] (3 args)");

BENCHMARK(macgen,
  macgen(a[i], b[i], c[i]);
, 8.*W<float>::c, "generic a[i] += b[i]*c[i]");

BENCHMARK(macstd,
  macstd(ac[i], bc[i], cc[i]);
, 8.*W<float>::c, "std a[i] += b[i]*c[i]");

BENCHMARK(macavx,
  macavxf(aavx[i], bavx[i], cavx[i]);
, 8.*W<float>::c, "AVX a[i] += b[i]*c[i]");

BENCHMARK(matmulgen,
  unsigned int os = i*nCoef;
	MATMUL(a, b, c, mulgen<float>, macgen<float>, os);
, nRow*nRow*6.*W<float>::c + nRow*nRow*(nRow-1)*8.*W<float>::c, 
  "generic A[i] = B[i]*C[i]");

BENCHMARK(matmulstd,
  unsigned int os = i*nCoef;
  MATMUL(ac, bc, cc, mulstd<float>, macstd<float>, os);
, nRow*nRow*6.*W<float>::c + nRow*nRow*(nRow-1)*8.*W<float>::c, 
  "std A[i] = B[i]*C[i]");

BENCHMARK(matmulavx,
  unsigned int os = i*nCoef;
  MATMUL(aavx, bavx, cavx, mulavxf, macavxf, os);
, nRow*nRow*6.*W<float>::c + nRow*nRow*(nRow-1)*8.*W<float>::c, 
  "AVX A[i] = B[i]*C[i]");

BENCHMARK(matmulunrollgen,
  unsigned int os = i*nCoef;
  MATMULUNROLL(a, b, c, mulgen<float>, macgen<float>, os);
, nRow*nRow*6.*W<float>::c + nRow*nRow*(nRow-1)*8.*W<float>::c, 
  "generic A[i] = B[i]*C[i] (unrolled)");

BENCHMARK(matmulunrollavx,
  unsigned int os = i*nCoef;
  MATMULUNROLL(aavx, bavx, cavx, mulavxf, macavxf, os);
, nRow*nRow*6.*W<float>::c + nRow*nRow*(nRow-1)*8.*W<float>::c, 
  "AVX A[i] = B[i]*C[i] (unrolled)");
