// use simdeez::*;

#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// DP high components and caller ignores returned high components
#[inline]
pub fn hi_dp_ss(a: __m128, b: __m128) -> __m128 {
    unsafe {
        // 0 1 2 3 -> 1 + 2 + 3, 0, 0, 0

        let mut res: __m128 = _mm_mul_ps(a, b);

        // 0 1 2 3 -> 1 1 3 3
        let hi = _mm_movehdup_ps(res);

        // 0 1 2 3 + 1 1 3 3 -> (0 + 1, 1 + 1, 2 + 3, 3 + 3)
        let sum = _mm_add_ps(hi, res);

        // unpacklo: 0 0 1 1
        res = _mm_add_ps(sum, _mm_unpacklo_ps(res, res));

        // (1 + 2 + 3, _, _, _)
        _mm_movehl_ps(res, res)
    }
}

/// Reciprocal with an additional single Newton-Raphson refinement
#[inline]
pub fn rcp_nr1(a: __m128) -> __m128 {
    //! ```f(x) = 1/x - a
    //! f'(x) = -1/x^2
    //! x_{n+1} = x_n - f(x)/f'(x)
    //!         = 2x_n - a x_n^2 = x_n (2 - a x_n)
    //! ```
    //! ~2.7x baseline with ~22 bits of accuracy
    unsafe {
        let xn: __m128 = _mm_rcp_ps(a);
        let axn: __m128 = _mm_mul_ps(a, xn);
        _mm_mul_ps(xn, _mm_sub_ps(_mm_set1_ps(2.0), axn))
    }
}

/// Reciprocal sqrt with an additional single Newton-Raphson refinement.
#[inline]
pub fn rsqrt_nr1(a: __m128) -> __m128 {
    //! ```f(x) = 1/x^2 - a
    //! f'(x) = -1/(2x^(3/2))
    //! Let x_n be the estimate, and x_{n+1} be the refinement
    //! x_{n+1} = x_n - f(x)/f'(x)
    //!         = 0.5 * x_n * (3 - a x_n^2)
    //!```
    //! From Intel optimization manual: expected performance is ~5.2x
    //! baseline (sqrtps + divps) with ~22 bits of accuracy

    unsafe {
        let xn = _mm_rsqrt_ps(a);
        let mut axn2 = _mm_mul_ps(xn, xn);
        axn2 = _mm_mul_ps(a, axn2);
        let xn3 = _mm_sub_ps(_mm_set1_ps(3.0), axn2);
        _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(0.5), xn), xn3)
    }
}

// Sqrt Newton-Raphson is evaluated in terms of rsqrt_nr1
#[inline]
pub fn sqrt_nr1(a: __m128) -> __m128 {
    unsafe { _mm_mul_ps(a, rsqrt_nr1(a)) }
}

#[cfg(target_feature = "sse4.1")]
#[inline]
pub fn hi_dp(a: __m128, b: __m128) -> __m128 {
    unsafe { _mm_dp_ps(a, b, 0b11100001) }
}

#[cfg(target_feature = "sse4.1")]
#[inline]
pub fn hi_dp_bc(a: __m128, b: __m128) -> __m128 {
    unsafe { _mm_dp_ps(a, b, 0b11101111) }
}

#[cfg(target_feature = "sse4.1")]
#[inline]
pub fn dp(a: __m128, b: __m128) -> __m128 {
    unsafe { _mm_dp_ps(a, b, 0b11110001) }
}

#[cfg(target_feature = "sse4.1")]
#[inline]
pub fn dp_bc(a: __m128, b: __m128) -> __m128 {
    unsafe { _mm_dp_ps(a, b, 0xff) }
}

#[cfg(not(target_feature = "sse4.1"))]
// // Equivalent to _mm_dp_ps(a, b, 0b11100001);
#[inline]
pub fn hi_dp(a: __m128, b: __m128) -> __m128 {
    // 0 1 2 3 -> 1 + 2 + 3, 0, 0, 0

    unsafe {
        let mut out = _mm_mul_ps(a, b);

        // 0 1 2 3 -> 1 1 3 3
        let hi = _mm_movehdup_ps(out);

        // 0 1 2 3 + 1 1 3 3 -> (0 + 1, 1 + 1, 2 + 3, 3 + 3)
        let sum = _mm_add_ps(hi, out);

        // unpacklo: 0 0 1 1
        out = _mm_add_ps(sum, _mm_unpacklo_ps(out, out));

        // (1 + 2 + 3, _, _, _)
        out = _mm_movehl_ps(out, out);

        _mm_and_ps(out, _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1)))
    }
}

#[cfg(not(target_feature = "sse4.1"))]
#[inline]
pub fn hi_dp_bc(a: __m128, b: __m128) -> __m128 {
    unsafe {
        // Multiply across and mask low component
        let mut out = _mm_mul_ps(a, b);

        // 0 1 2 3 -> 1 1 3 3
        let hi = _mm_movehdup_ps(out);

        // 0 1 2 3 + 1 1 3 3 -> (0 + 1, 1 + 1, 2 + 3, 3 + 3)
        let sum = _mm_add_ps(hi, out);

        // unpacklo: 0 0 1 1
        out = _mm_add_ps(sum, _mm_unpacklo_ps(out, out));

        // port comment: see https://github.com/Ziriax/KleinSharp/blob/master/KleinSharp.ConstSwizzleTool/Program.cs
        _mm_shuffle_ps(out, out, (2 << 6) | (2 << 4) | (2 << 2) | 2)
    }
}

#[cfg(not(target_feature = "sse4.1"))]
#[inline]
pub fn dp(a: __m128, b: __m128) -> __m128 {
    unsafe {
        // Multiply across and shift right (shifting in zeros)
        let mut out = _mm_mul_ps(a, b);
        let hi = _mm_movehdup_ps(out);
        // (a1 b1, a2 b2, a3 b3, 0) + (a2 b2, a2 b2, 0, 0)
        // = (a1 b1 + a2 b2, _, a3 b3, 0)
        out = _mm_add_ps(hi, out);
        out = _mm_add_ss(out, _mm_movehl_ps(hi, out));
        _mm_and_ps(out, _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1)))
    }
}

#[cfg(not(target_feature = "sse4.1"))]
#[inline]
pub fn dp_bc(a: __m128, b: __m128) -> __m128 {
    unsafe {
        // Multiply across and shift right (shifting in zeros)
        let mut out = _mm_mul_ps(a, b);
        let hi = _mm_movehdup_ps(out);
        // (a1 b1, a2 b2, a3 b3, 0) + (a2 b2, a2 b2, 0, 0)
        // = (a1 b1 + a2 b2, _, a3 b3, 0)
        out = _mm_add_ps(hi, out);
        out = _mm_add_ss(out, _mm_movehl_ps(hi, out));
        // port comment: see https://github.com/Ziriax/KleinSharp/blob/master/KleinSharp.ConstSwizzleTool/Program.cs
        //	    return KLN_SWIZZLE(out, 0, 0, 0, 0);
        _mm_shuffle_ps(out, out, 0)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    #[test]
    fn rcp_nr1() {
        let mut buf: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

        unsafe {
            let a: __m128 = _mm_set_ps(4.0, 3.0, 2.0, 1.0);

            let b: __m128 = super::rcp_nr1(a);

            _mm_store_ps(&mut (buf[0]), b);
        }

        approx_eq(buf[0], 1.0);
        approx_eq(buf[1], 0.5);
        approx_eq(buf[2], 1.0 / 3.0);
        approx_eq(buf[3], 0.25);
    }
}
