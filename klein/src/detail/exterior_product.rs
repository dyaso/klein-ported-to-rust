#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::sse::{dp, hi_dp};

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

#[inline]
pub fn ext00(a: __m128, b: __m128, p1Out: &mut __m128, p2Out: &mut __m128) {
    // (a1 b2 - a2 b1) e12 +
    // (a2 b3 - a3 b2) e23 +
    // (a3 b1 - a1 b3) e31 +

    // (a0 b1 - a1 b0) e01 +
    // (a0 b2 - a2 b0) e02 +
    // (a0 b3 - a3 b0) e03

    unsafe {
        *p1Out = _mm_mul_ps(a, _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */));
        let um = _mm_sub_ps(
            *p1Out,
            _mm_mul_ps(_mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */), b),
        );
        *p1Out = _mm_shuffle_ps(um, um, 120 /* 1, 3, 2, 0*/);

        *p2Out = _mm_mul_ps(_mm_shuffle_ps(a, a, 0 /* 0, 0, 0, 0 */), b);
        *p2Out = _mm_sub_ps(
            *p2Out,
            _mm_mul_ps(a, _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */)),
        );
    }

    // For both outputs above, we don't zero the lowest component because
    // we've arranged a cancellation
}

// Plane ^ Branch (branch is a line through the origin)
#[inline]
pub fn extPB(a: __m128, b: __m128) -> __m128 {
    // (a1 b1 + a2 b2 + a3 b3) e123 +
    // (-a0 b1) e032 +
    // (-a0 b2) e013 +
    // (-a0 b3) e021

    unsafe {
        let mut p3Out = _mm_mul_ps(
            _mm_mul_ps(_mm_shuffle_ps(a, a, 1 /* 0, 0, 0, 1 */), b),
            _mm_set_ps(-1., -1., -1., 0.),
        );

        return _mm_add_ss(p3Out, hi_dp(a, b));
    }
}

// p0 ^ p2 = p2 ^ p0
pub fn ext02(a: __m128, b: __m128) -> __m128 {
    // (a1 b2 - a2 b1) e021
    // (a2 b3 - a3 b2) e032 +
    // (a3 b1 - a1 b3) e013 +

    unsafe {
        let mut p3Out = _mm_mul_ps(a, _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */));
        let um = _mm_sub_ps(
            p3Out,
            _mm_mul_ps(_mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */), b),
        );
        p3Out = _mm_shuffle_ps(
            um, um,
            //_mm_sub_ps(p3Out, _mm_mul_ps(_mm_shuffle_ps(a,a, 120 /* 1, 3, 2, 0 */), b)),
            120, /* 1, 3, 2, 0 */
        );

        return p3Out;
    }
}

#[inline]
pub fn ext03_false(a: __m128, b: __m128) -> __m128 {
    // (a0 b0 + a1 b1 + a2 b2 + a3 b3) e0123
    //   p2 =
    dp(a, b)
}

#[inline]
pub fn ext03_true(a: __m128, b: __m128) -> __m128 {
    let mut p2 = dp(a, b);
    unsafe { return _mm_xor_ps(p2, _mm_set_ss(-0.)) }
}

// [MethodImpl(MethodImplOptions.AggressiveInlining)]
// public static __m128 ext03(bool flip, __m128 a, __m128 b)
// {
// 	// (a0 b0 + a1 b1 + a2 b2 + a3 b3) e0123
// 	var p2 = dp(a, b);

// 	if (flip)
// 	{
// 		// p0 ^ p3 = -p3 ^ p0
// 		p2 = _mm_xor_ps(p2, _mm_set_ss(-0f));
// 	}

// 	return p2;
// }

// // The exterior products p2 ^ p2, p2 ^ p3, p3 ^ p2, and p3 ^ p3 all vanish
// }
// }
