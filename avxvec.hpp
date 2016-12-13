#ifndef CMUL_AVX_HPP_
#define CMUL_AVX_HPP_

#include <immintrin.h>

#ifndef strong_inline
#define strong_inline __attribute__((always_inline))
#endif

#define SELECT(A,B,C,D) ((A<<6)|(B<<4)|(C<<2)|(D))

strong_inline __m256 muladdf(const __m256 &a, const __m256 &b)
{
	return _mm256_add_ps(a, b);
}

strong_inline __m256 mulavxf(const __m256 &a, const __m256 &b)
{
  __m256 a_real = _mm256_moveldup_ps(a);
  __m256 a_imag = _mm256_movehdup_ps(a);

  a_imag = _mm256_mul_ps(a_imag, _mm256_shuffle_ps(b,b, SELECT(2,3,0,1)));

  return _mm256_fmaddsub_ps(a_real, b, a_imag);
}

strong_inline void mulavxf(__m256 &a, const __m256 &b, const __m256 &c)
{
	a = mulavxf(b, c);
}

strong_inline void macavxf(__m256 &a, const __m256 &b, const __m256 &c)
{
	a = muladdf(a, mulavxf(b, c));
}

#undef SELECT

#endif
