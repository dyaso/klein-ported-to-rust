#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::sse::{hi_dp, hi_dp_ss};

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

#[inline]
pub fn dot00(a: __m128, b: __m128) -> __m128 {
    // a1 b1 + a2 b2 + a3 b3
    // p1_out =
    return hi_dp(a, b);
}

// The symmetric inner product on these two partitions commutes
#[inline]
pub fn dot03(a: __m128, b: __m128, p1_out: &mut __m128, p2_out: &mut __m128) {
    // (a2 b1 - a1 b2) e03 +
    // (a3 b2 - a2 b3) e01 +
    // (a1 b3 - a3 b1) e02 +
    // a1 b0 e23 +
    // a2 b0 e31 +
    // a3 b0 e12
    unsafe {
        *p1_out = _mm_mul_ps(a, _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */));

        if is_x86_feature_detected!("sse4.1") {
            *p1_out = _mm_blend_ps(*p1_out, _mm_setzero_ps(), 1);
        } else {
            *p1_out = _mm_and_ps(*p1_out, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)));
        }

        let um = _mm_sub_ps(
            _mm_mul_ps(_mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */), b),
            _mm_mul_ps(a, _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */)),
        );
        *p2_out = _mm_shuffle_ps(um, um, 120 /* 1, 3, 2, 0 */);
    }
}

#[inline]
pub fn dot11(a: __m128, b: __m128) -> __m128 {
    unsafe { return _mm_xor_ps(_mm_set_ss(-0.), hi_dp_ss(a, b)) }
}

#[inline]
pub fn dot33(a: __m128, b: __m128) -> __m128 {
    // -a0 b0
    unsafe { return _mm_mul_ps(_mm_set_ss(-1.), _mm_mul_ss(a, b)) }
}

// Point | Line
#[inline]
pub fn dot_ptl(a: __m128, b: __m128) -> __m128 {
    // (a1 b1 + a2 b2 + a3 b3) e0 +
    // -a0 b1 e1 +
    // -a0 b2 e2 +
    // -a0 b3 e3

    unsafe {
        let dp = hi_dp_ss(a, b);
        let mut p0 = _mm_mul_ps(_mm_shuffle_ps(a, a, 0), b);
        p0 = _mm_xor_ps(p0, _mm_set_ps(-0., -0., -0., 0.));

        if is_x86_feature_detected!("sse4.1") {
            return _mm_blend_ps(p0, dp, 1);
        } else {
            return _mm_add_ss(p0, dp);
        }
    }
}

// Plane | Line
#[inline]
pub fn dot_pl_noflip(a: __m128, b: __m128, c: __m128) -> __m128 {
    // -(a1 c1 + a2 c2 + a3 c3) e0 +
    // (a2 b1 - a1 b2) e3
    // (a3 b2 - a2 b3) e1 +
    // (a1 b3 - a3 b1) e2 +
    unsafe {
        let mut p0 = _mm_mul_ps(_mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */), b);
        p0 = _mm_sub_ps(
            p0,
            _mm_mul_ps(a, _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */)),
        );
        return _mm_sub_ss(_mm_shuffle_ps(p0, p0, 120 /* 1, 3, 2, 0 */), hi_dp_ss(a, c));
    }
}

#[inline]
pub fn dot_pl_flip(a: __m128, b: __m128, c: __m128) -> __m128 {
    // (a1 c1 + a2 c2 + a3 c3) e0 +
    // (a1 b2 - a2 b1) e3
    // (a2 b3 - a3 b2) e1 +
    // (a3 b1 - a1 b3) e2 +

    unsafe {
        let mut p0 = _mm_mul_ps(a, _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */));
        p0 = _mm_sub_ps(
            p0,
            _mm_mul_ps(_mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */), b),
        );
        return _mm_add_ss(_mm_shuffle_ps(p0, p0, 120 /* 1, 3, 2, 0 */), hi_dp_ss(a, c));
    }
}

// Plane | Ideal Line
#[inline]
pub fn dot_pil_flip(a: __m128, c: __m128) -> __m128 {
    hi_dp(a, c)
}

#[inline]
pub fn dot_pil_noflip(a: __m128, c: __m128) -> __m128 {
    let p0 = hi_dp(a, c);
    unsafe { return _mm_xor_ps(p0, _mm_set_ss(-0.)) }
}
