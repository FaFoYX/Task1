/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_FUNCTIONS_HEADER
#define MATH_FUNCTIONS_HEADER

#include <bitset>
#include <cmath>
#include <cinttypes>

#ifdef _MSC_VER
#   include <intrin.h>
#endif

#include "math/defines.h"
#include "math/vector.h"
#include "math/matrix.h"

namespace math {


    template <typename T>
    inline T
        gaussian(T const& x, T const& sigma)
    {
        return std::exp(-((x * x) / (T(2) * sigma * sigma)));
    }

    template <typename T>
    inline T
        gaussian_xx(T const& xx, T const& sigma)
    {
        return std::exp(-(xx / (T(2) * sigma * sigma)));
    }

    template <typename T>
    inline T
        gaussian_2d(T const& x, T const& y, T const& sigma_x, T const& sigma_y)
    {
        return std::exp(-(x * x) / (T(2) * sigma_x * sigma_x)
            - (y * y) / (T(2) * sigma_y * sigma_y));
    }

    template <typename T>
    inline T
        round(T const& x)
    {
        return x > T(0) ? std::floor(x + T(0.5)) : std::ceil(x - T(0.5));
    }

   
    template <typename T>
    inline T
        interpolate(T const& v1, float w1)
    {
        return v1 * w1;
    }

    template <>
    inline unsigned char
        interpolate(unsigned char const& v1, float w1)
    {
        return (unsigned char)((float)v1 * w1 + 0.5f);
    }

 
    template <typename T>
    inline T
        interpolate(T const& v1, T const& v2, float w1, float w2)
    {
        return v1 * w1 + v2 * w2;
    }

    template <>
    inline unsigned char
        interpolate(unsigned char const& v1, unsigned char const& v2,
            float w1, float w2)
    {
        return (unsigned char)((float)v1 * w1 + (float)v2 * w2 + 0.5f);
    }

    template <typename T>
    inline T
        interpolate(T const& v1, T const& v2, T const& v3,
            float w1, float w2, float w3)
    {
        return v1 * w1 + v2 * w2 + v3 * w3;
    }

    template <>
    inline unsigned char
        interpolate(unsigned char const& v1, unsigned char const& v2,
            unsigned char const& v3, float w1, float w2, float w3)
    {
        return (unsigned char)((float)v1 * w1 + (float)v2 * w2
            + (float)v3 * w3 + 0.5f);
    }


    template <typename T>
    inline T
        interpolate(T const& v1, T const& v2, T const& v3, T const& v4,
            float w1, float w2, float w3, float w4)
    {
        return v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4;
    }

 
    template <>
    inline unsigned char
        interpolate(unsigned char const& v1, unsigned char const& v2,
            unsigned char const& v3, unsigned char const& v4,
            float w1, float w2, float w3, float w4)
    {
        return (unsigned char)((float)v1 * w1 + (float)v2 * w2
            + (float)v3 * w3 + (float)v4 * w4 + 0.5f);
    }

    template <typename T>
    inline T
        sinc(T const& x)
    {
        return (x == T(0) ? T(1) : std::sin(x) / x);
    }

    template <typename T>
    std::size_t constexpr
        popcount(T const x)
    {
        static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value,
            "T must be an unsigned integral type.");
        return std::bitset<sizeof(T) * 8>(x).count();
    }

#ifdef _MSC_VER // MSVS fails to optimize std::bitset::count
    std::size_t inline
        popcount(std::uint32_t const x)
    {
#   if !(__amd64__ || __x86_64__ || _WIN64 || _M_X64)
        __pragma (message("Warning: CPU support for _mm_popcnt_u32 is assumed, "
            "but could not be verified."))
#   endif
            return static_cast<std::size_t>(_mm_popcnt_u32(x));
    }

    std::size_t inline
        popcount(std::uint64_t const x)
    {
#   if __amd64__ || __x86_64__ || _WIN64 || _M_X64
        return static_cast<std::size_t>(_mm_popcnt_u64(x));
#   else
        std::uint32_t const low_bits = static_cast<std::uint32_t>(x);
        std::uint32_t const high_bits = static_cast<std::uint32_t>(x >> 32);
        return popcount(low_bits) + popcount(high_bits);
#   endif
    }
#endif

 
    template <typename T>
    T const&
        clamp(T const& v, T const& min = T(0), T const& max = T(1))
    {
        return (v < min ? min : (v > max ? max : v));
    }

    template <typename T>
    T
        bound_mirror(T const& v, T const& min, T const& max)
    {
        return (v < min ? min - v : (v > max ? max + max - v : v));
    }

    template <typename T>
    T const& min(T const& a, T const& b, T const& c)
    {
        return (std::min)(a, (std::min)(b, c));
    }

    template <typename T>
    T const& max(T const& a, T const& b, T const& c)
    {
        return (std::max)(a, (std::max)(b, c));
    }

    template <typename T>
    T
        fastpow(T const& base, unsigned int exp)
    {
        T ret = (exp == 0 ? T(1) : base);
        unsigned int const exp2 = exp / 2;
        unsigned int i = 1;
        for (; i <= exp2; i = i * 2)
            ret = ret * ret;
        for (; i < exp; ++i)
            ret = ret * base;
        return ret;
    }

    inline int
        to_gray_code(int bin)
    {
        return bin ^ (bin >> 1);
    }

    inline int
        from_gray_code(int gc)
    {
        int b = 0, ret = 0;
        for (int i = 31; i >= 0; --i)
        {
            b = (b + ((gc & (1 << i)) >> i)) % 2;
            ret |= b << i;
        }
        return ret;
    }
}

#endif // MATH_FUNCTIONS_HEADER
