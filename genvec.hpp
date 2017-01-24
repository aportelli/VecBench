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

#ifndef CMUL_GENVEC_HPP_
#define CMUL_GENVEC_HPP_

#include <complex>
#include <iostream>

#ifndef strong_inline
#define strong_inline __attribute__((always_inline)) inline
#endif

// generic SIMD length
#define GEN_SIMD_DCOMPLEX_WIDTH 2u
constexpr unsigned int w   = GEN_SIMD_DCOMPLEX_WIDTH*16u;

// macros to manage compiler specific pragmas
//#define VECTOR_LOOPS

#ifdef VECTOR_LOOPS
#ifdef __INTEL_COMPILER
#define VECTOR_FOR(i, w, inc)\
_Pragma("simd vectorlength(w*8)")\
for (unsigned int i = 0; i < w; i += inc)
#elif defined __clang__
#define VECTOR_FOR(i, w, inc)\
_Pragma("clang loop unroll(full) vectorize(enable) interleave(enable) vectorize_width(w)")\
for (unsigned int i = 0; i < w; i += inc)
#else
#define VECTOR_FOR(i, w, inc)\
for (unsigned int i = 0; i < w; i += inc)
#endif
#else
#define VECTOR_FOR(i, w, inc)\
for (unsigned int i = 0; i < w; i += inc)
#endif

// generic vector type
template <typename T> struct W;
template <> struct W<double> {
  constexpr static unsigned int c = GEN_SIMD_DCOMPLEX_WIDTH;
  constexpr static unsigned int r = 2*c;
};
template <> struct W<float> {
  constexpr static unsigned int c = 2*W<double>::c;
  constexpr static unsigned int r = 2*c;
};

template <typename T> 
struct vec {
  alignas(w) T v[W<T>::r];
};

template <typename T>
struct vecc {
  alignas(w) std::complex<T> v[W<T>::c];
};

typedef vec<float>   vecf;
typedef vec<double>  vecd;
typedef vecc<float>  vecfc;
typedef vecc<double> vecdc;

// IO
template <typename T>
strong_inline std::ostream & operator<<(std::ostream & out, const vec<T> & a)
{
  unsigned int i;

  out << "[";
  for (i = 0; i < W<T>::r - 1; ++i)
  {
    out << a.v[i] << ", "; 
  }
  out << a.v[i] << "]";

  return out;
}

template <typename T>
strong_inline std::ostream & operator<<(std::ostream & out, const vecc<T> & a)
{
  unsigned int i;

  out << "[";
  for (i = 0; i < W<T>::c - 1; ++i)
  {
    out << a.v[i].real() << ", " << a.v[i].imag() << ", "; 
  }
  out << a.v[i].real() << ", " << a.v[i].imag() << "]";

  return out;
}

// naive add
template <typename T>
strong_inline vec<T> add(const vec<T> &a, const vec<T> &b){
  vec<T> out;

  VECTOR_FOR(i, W<T>::r, 1)
  {
    out.v[i] = a.v[i] + b.v[i];
  }      
  
  return out;
}

// naive complex mutliply
#define cmul(a, b, c, i)\
a[i]   = b[i]*c[i]   - b[i+1]*c[i+1];\
a[i+1] = b[i]*c[i+1] + b[i+1]*c[i];

template <typename T>
strong_inline vec<T> mulgen(const vec<T> &a, const vec<T> &b){
  vec<T> out;

  VECTOR_FOR(i, W<T>::c, 1)
  {
    cmul(out.v, a.v, b.v, 2*i);
  }      
  
  return out;
}

template <typename T>
strong_inline void mulgen(vec<T> &a, const vec<T> &b, const vec<T> &c){
  VECTOR_FOR(i, W<T>::c, 1)
  {
    cmul(a.v, b.v, c.v, 2*i);
  }      
}

#undef cmul

// naive complex accumulate
#define cmac(a, b, c, i)\
a[i]   += b[i]*c[i]   - b[i+1]*c[i+1];\
a[i+1] += b[i]*c[i+1] + b[i+1]*c[i];

template <typename T>
strong_inline void macgen(vec<T> &a, const vec<T> &b, const vec<T> &c){
  VECTOR_FOR(i, W<T>::c, 1)
  {
    cmac(a.v, b.v, c.v, 2*i);
  }      
}

#undef cmac

// complex multiply using the standard library
template <typename T>
strong_inline vecc<T> mulstd(const vecc<T> &a, const vecc<T> &b)
{
  vecc<T> c;
  
  VECTOR_FOR(i, W<T>::c, 1)
  {
    c.v[i] = a.v[i]*b.v[i];
  }

  return c;
}

template <typename T>
strong_inline void mulstd(vecc<T> &a, const vecc<T> &b, const vecc<T> &c)
{
  VECTOR_FOR(i, W<T>::c, 1)
  {
    a.v[i] = b.v[i]*c.v[i];
  }
}

template <typename T>
strong_inline void macstd(vecc<T> &a, const vecc<T> &b, const vecc<T> &c)
{
  VECTOR_FOR(i, W<T>::c, 1)
  {
    a.v[i] += b.v[i]*c.v[i];
  }
}

// construct -i
template <typename T>
strong_inline vec<T> mi(void){
  vec<T> out;

  VECTOR_FOR(i, W<T>::c, 1)
  {
    out.v[2*i]   = 0.;
    out.v[2*i+1] = -1.;
  }      
  
  return out;
}

// mutliply by -i explicitly
#define tmi(a, c, i)\
c[i]   = a[i+1];\
c[i+1] = -a[i];

template <typename T>
strong_inline vec<T> timesMinusI1(const vec<T> &a){
  vec<T> out;

  VECTOR_FOR(i, W<T>::c, 1)
  {
    tmi(a.v, out.v, 2*i);
  }      
  
  return out;
}

#undef cmul

// mutliply by -i by defining -i as a constant and using mulgen routine
static const vecf mif = mi<float>();
static const vecd mid = mi<double>();
template <typename T> struct micst;
template <> struct micst<float>  {static constexpr const vecf *val = &mif;};
template <> struct micst<double> {static constexpr const vecd *val = &mid;};

template <typename T>
strong_inline vec<T> timesMinusI2(const vec<T> &a){
  return mulgen(a, *micst<T>::val);
}

#endif
