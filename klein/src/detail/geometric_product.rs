#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{dp, rcp_nr1}; //hi_dp, hi_dp_bc, hi_dp_ss, rsqrt_nr1

// Purpose: Define functions of the form gpAB where A and B are partition
// indices. Each function so-defined computes the geometric product using vector
// intrinsics. The partition index determines which basis elements are present
// in each XMM component of the operand.
// A number of the computations in this file are performed symbolically in
// scripts/validation.klein

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e12, e31, e23)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)

#[inline]
pub fn gp00(a: __m128, b: __m128, p1_out: &mut __m128, p2_out: &mut __m128) {
    // (a1 b1 + a2 b2 + a3 b3) +

    // (a2 b3 - a3 b2) e23 +
    // (a3 b1 - a1 b3) e31 +
    // (a1 b2 - a2 b1) e12 +

    // (a0 b1 - a1 b0) e01 +
    // (a0 b2 - a2 b0) e02 +
    // (a0 b3 - a3 b0) e03

    unsafe {
        *p1_out = _mm_mul_ps(
            _mm_shuffle_ps(a, a, 121 /* 1, 3, 2, 1 */),
            _mm_shuffle_ps(b, b, 157 /* 2, 1, 3, 1 */),
        );

        *p1_out = _mm_sub_ps(
            *p1_out,
            _mm_xor_ps(
                _mm_set_ss(-0.),
                _mm_mul_ps(
                    _mm_shuffle_ps(a, a, 158 /* 2, 1, 3, 2 */),
                    _mm_shuffle_ps(b, b, 122 /* 1, 3, 2, 2 */),
                ),
            ),
        );
        // Add a3 b3 to the lowest component
        *p1_out = _mm_add_ss(
            *p1_out,
            _mm_mul_ps(
                _mm_shuffle_ps(a, a, 3 /* 0, 0, 0, 3 */),
                _mm_shuffle_ps(b, b, 3 /* 0, 0, 0, 3 */),
            ),
        );

        // (a0 b0, a0 b1, a0 b2, a0 b3)
        *p2_out = _mm_mul_ps(_mm_shuffle_ps(a, a, 0 /* 0, 0, 0, 0 */), b);

        // Sub (a0 b0, a1 b0, a2 b0, a3 b0)
        // Note that the lowest component cancels
        *p2_out = _mm_sub_ps(
            *p2_out,
            _mm_mul_ps(a, _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */)),
        );
    }
}

// p0: (e0, e1, e2, e3)
// p3: (e123, e032, e013, e021)
// p1: (1, e12, e31, e23)
// p2: (e0123, e01, e02, e03)
#[inline]
pub fn gp03_flip(a: __m128, b: __m128, p1: &mut __m128, p2: &mut __m128) {
    // a1 b0 e23 +
    // a2 b0 e31 +
    // a3 b0 e12 +
    // (a0 b0 + a1 b1 + a2 b2 + a3 b3) e0123 +
    // (a3 b2 - a2 b3) e01 +
    // (a1 b3 - a3 b1) e02 +
    // (a2 b1 - a1 b2) e03
    unsafe {
        *p1 = _mm_mul_ps(a, _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */));
        if is_x86_feature_detected!("sse4.1") {
            *p1 = _mm_blend_ps(*p1, _mm_setzero_ps(), 1);
        } else {
            *p1 = _mm_and_ps(*p1, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)));
        }

        // (_, a3 b2, a1 b3, a2 b1)
        *p2 = _mm_mul_ps(
            _mm_shuffle_ps(a, a, 156 /* 2, 1, 3, 0 */),
            _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */),
        );
        *p2 = _mm_sub_ps(
            *p2,
            _mm_mul_ps(
                _mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */),
                _mm_shuffle_ps(b, b, 156 /* 2, 1, 3, 0 */),
            ),
        );

        // Compute a0 b0 + a1 b1 + a2 b2 + a3 b3 and store it in the low
        // component
        let mut tmp = dp(a, b);

        tmp = _mm_xor_ps(tmp, _mm_set_ss(-0.));

        *p2 = _mm_add_ps(*p2, tmp);
    }
}

pub fn gp03_noflip(a: __m128, b: __m128, p1: &mut __m128, p2: &mut __m128) {
    unsafe {
        // a1 b0 e23 +
        // a2 b0 e31 +
        // a3 b0 e12 +
        // -(a0 b0 + a1 b1 + a2 b2 + a3 b3) e0123 +
        // (a3 b2 - a2 b3) e01 +
        // (a1 b3 - a3 b1) e02 +
        // (a2 b1 - a1 b2) e03

        *p1 = _mm_mul_ps(a, _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */));
        if is_x86_feature_detected!("sse4.1") {
            *p1 = _mm_blend_ps(*p1, _mm_setzero_ps(), 1);
        } else {
            *p1 = _mm_and_ps(*p1, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)));
        }

        // (_, a3 b2, a1 b3, a2 b1)
        *p2 = _mm_mul_ps(
            _mm_shuffle_ps(a, a, 156 /* 2, 1, 3, 0 */),
            _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */),
        );
        *p2 = _mm_sub_ps(
            *p2,
            _mm_mul_ps(
                _mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */),
                _mm_shuffle_ps(b, b, 156 /* 2, 1, 3, 0 */),
            ),
        );

        // Compute a0 b0 + a1 b1 + a2 b2 + a3 b3 and store it in the low
        // component
        let tmp = dp(a, b);
        *p2 = _mm_add_ps(*p2, tmp);
    }
}

#[inline]
pub fn gp11(a: __m128, b: __m128, p1_out: &mut __m128) {
    // (a0 b0 - a1 b1 - a2 b2 - a3 b3) +
    // (a0 b1 - a2 b3 + a1 b0 + a3 b2)*e23
    // (a0 b2 - a3 b1 + a2 b0 + a1 b3)*e31
    // (a0 b3 - a1 b2 + a3 b0 + a2 b1)*e12

    // We use abcd to refer to the slots to avoid conflating bivector/scalar
    // coefficients with cartesian coordinates

    // In general, we can get rid of at most one swizzle
    unsafe {
        *p1_out = _mm_mul_ps(_mm_shuffle_ps(a, a, 0 /*0, 0, 0, 0*/), b);

        *p1_out = _mm_sub_ps(
            *p1_out,
            _mm_mul_ps(
                _mm_shuffle_ps(a, a, 121 /* 1, 3, 2, 1 */),
                _mm_shuffle_ps(b, b, 157 /* 2, 1, 3, 1 */),
            ),
        );

        // In a separate register, accumulate the later components so we can
        // negate the lower single-precision element with a single instruction
        let tmp1 = _mm_mul_ps(
            _mm_shuffle_ps(a, a, 230 /* 3, 2, 1, 2 */),
            _mm_shuffle_ps(b, b, 2 /*0, 0, 0, 2*/),
        );

        let tmp2 = _mm_mul_ps(
            _mm_shuffle_ps(a, a, 159 /*2, 1, 3, 3*/),
            _mm_shuffle_ps(b, b, 123 /*1, 3, 2, 3*/),
        );

        let tmp = _mm_xor_ps(_mm_add_ps(tmp1, tmp2), _mm_set_ss(-0.));

        *p1_out = _mm_add_ps(*p1_out, tmp);
    }
}

// p3: (e123, e021, e013, e032) // p2: (e0123, e01, e02, e03) [MethodImpl(MethodImplOptions.AggressiveInlining)]
#[inline]
pub fn gp33(a: __m128, b: __m128) -> __m128 {
    // (-a0 b0) +
    // (-a0 b1 + a1 b0) e01 +
    // (-a0 b2 + a2 b0) e02 +
    // (-a0 b3 + a3 b0) e03
    //
    // Produce a translator by dividing all terms by a0 b0

    unsafe {
        let mut tmp = _mm_mul_ps(_mm_shuffle_ps(a, a, 0 /* 0, 0, 0, 0 */), b);
        tmp = _mm_mul_ps(tmp, _mm_set_ps(-1., -1., -1., -2.));
        tmp = _mm_add_ps(tmp, _mm_mul_ps(a, _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */)));

        // (0, 1, 2, 3) -> (0, 0, 2, 2)
        let mut ss = _mm_moveldup_ps(tmp);
        ss = _mm_movelh_ps(ss, ss);
        tmp = _mm_mul_ps(tmp, rcp_nr1(ss));

        if is_x86_feature_detected!("sse4.1") {
            _mm_blend_ps(tmp, _mm_setzero_ps(), 1)
        } else {
            _mm_and_ps(tmp, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)))
        }
    }
}

// [MethodImpl(MethodImplOptions.AggressiveInlining)]

// public static void gp_dl(float u, float v, __m128 b, __m128 c, out __m128 p1, out __m128 p2)
#[inline]
pub fn gp_dl(u: f32, v: f32, b: __m128, c: __m128, p1: &mut __m128, p2: &mut __m128) {
    // b1 u e23 +
    // b2 u e31 +
    // b3 u e12 +
    // (-b1 v + c1 u) e01 +
    // (-b2 v + c2 u) e02 +
    // (-b3 v + c3 u) e03
    unsafe {
        let u_vec = _mm_set1_ps(u);
        let v_vec = _mm_set1_ps(v);
        *p1 = _mm_mul_ps(u_vec, b);
        *p2 = _mm_mul_ps(c, u_vec);
        *p2 = _mm_sub_ps(*p2, _mm_mul_ps(b, v_vec));
    }
}

#[inline]
pub fn gp_rt(flip: bool, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let mut p2;

        if flip {
            // (a1 b1 + a2 b2 + a3 b3) e0123 +
            // (a0 b1 + a2 b3 - a3 b2) e01 +
            // (a0 b2 + a3 b1 - a1 b3) e02 +
            // (a0 b3 + a1 b2 - a2 b1) e03

            p2 = _mm_mul_ps(
                _mm_shuffle_ps(a, a, 1 /* 0, 0, 0, 1 */),
                _mm_shuffle_ps(b, b, 229 /* 3, 2, 1, 1 */),
            );
            p2 = _mm_add_ps(
                p2,
                _mm_mul_ps(
                    _mm_shuffle_ps(a, a, 122 /* 1, 3, 2, 2 */),
                    _mm_shuffle_ps(b, b, 158 /* 2, 1, 3, 2 */),
                ),
            );
            p2 = _mm_sub_ps(
                p2,
                _mm_xor_ps(
                    _mm_set_ss(-0.),
                    _mm_mul_ps(
                        _mm_shuffle_ps(a, a, 159 /*2, 1, 3, 3*/),
                        _mm_shuffle_ps(b, b, 123 /*1, 3, 2, 3*/),
                    ),
                ),
            );
        } else {
            // (a1 b1 + a2 b2 + a3 b3) e0123 +
            // (a0 b1 + a3 b2 - a2 b3) e01 +
            // (a0 b2 + a1 b3 - a3 b1) e02 +
            // (a0 b3 + a2 b1 - a1 b2) e03

            p2 = _mm_mul_ps(
                _mm_shuffle_ps(a, a, 1 /* 0, 0, 0, 1 */),
                _mm_shuffle_ps(b, b, 229 /* 3, 2, 1, 1 */),
            );
            p2 = _mm_add_ps(
                p2,
                _mm_mul_ps(
                    _mm_shuffle_ps(a, a, 158 /* 2, 1, 3, 2 */),
                    _mm_shuffle_ps(b, b, 122 /* 1, 3, 2, 2 */),
                ),
            );
            p2 = _mm_sub_ps(
                p2,
                _mm_xor_ps(
                    _mm_set_ss(-0.),
                    _mm_mul_ps(
                        _mm_shuffle_ps(a, a, 123 /*1, 3, 2, 3 */),
                        _mm_shuffle_ps(b, b, 159 /*2, 1, 3, 3*/),
                    ),
                ),
            );
        }

        p2
    }
}

#[inline]
pub fn gp12(flip: bool, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let mut p2 = gp_rt(flip, a, b);
        p2 = _mm_sub_ps(
            p2,
            _mm_xor_ps(
                _mm_set_ss(-0.),
                _mm_mul_ps(a, _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */)),
            ),
        );
        p2
    }
}

/// <summary>
/// Optimized line * line operation
/// (a,d) = first line P1, P2
/// (b,c) = second line P1, P2
/// </summary>
#[inline]
pub fn gp_ll(a: __m128, d: __m128, b: __m128, c: __m128, p1: &mut __m128, p2: &mut __m128) {
    // (-a1 b1 - a3 b3 - a2 b2) +
    // (a2 b1 - a1 b2) e12 +
    // (a1 b3 - a3 b1) e31 +
    // (a3 b2 - a2 b3) e23 +
    // (a1 c1 + a3 c3 + a2 c2 + b1 d1 + b3 d3 + b2 d2) e0123
    // (a3 c2 - a2 c3         + b2 d3 - b3 d2) e01 +
    // (a1 c3 - a3 c1         + b3 d1 - b1 d3) e02 +
    // (a2 c1 - a1 c2         + b1 d2 - b2 d1) e03 +
    unsafe {
        let flip = _mm_set_ss(-0.);

        *p1 = _mm_mul_ps(
            _mm_shuffle_ps(a, a, 217 /* 3, 1, 2, 1 */),
            _mm_shuffle_ps(b, b, 181 /* 2, 3, 1, 1 */),
        );
        *p1 = _mm_xor_ps(*p1, flip);
        *p1 = _mm_sub_ps(
            *p1,
            _mm_mul_ps(
                _mm_shuffle_ps(a, a, 183 /* 2, 3, 1, 3 */),
                _mm_shuffle_ps(b, b, 219 /* 3, 1, 2, 3 */),
            ),
        );
        let a2 = _mm_unpackhi_ps(a, a);
        let b2 = _mm_unpackhi_ps(b, b);
        *p1 = _mm_sub_ss(*p1, _mm_mul_ss(a2, b2));

        *p2 = _mm_mul_ps(
            _mm_shuffle_ps(a, a, 157 /* 2, 1, 3, 1 */),
            _mm_shuffle_ps(c, c, 121 /* 1, 3, 2, 1 */),
        );
        *p2 = _mm_sub_ps(
            *p2,
            _mm_xor_ps(
                flip,
                _mm_mul_ps(
                    _mm_shuffle_ps(a, a, 123 /*1, 3, 2, 3 */),
                    _mm_shuffle_ps(c, c, 159 /*2, 1, 3, 3*/),
                ),
            ),
        );
        *p2 = _mm_add_ps(
            *p2,
            _mm_mul_ps(
                _mm_shuffle_ps(b, b, 121 /* 1, 3, 2, 1 */),
                _mm_shuffle_ps(d, d, 157 /* 2, 1, 3, 1 */),
            ),
        );
        *p2 = _mm_sub_ps(
            *p2,
            _mm_xor_ps(
                flip,
                _mm_mul_ps(
                    _mm_shuffle_ps(b, b, 159 /*2, 1, 3, 3*/),
                    _mm_shuffle_ps(d, d, 123 /*1, 3, 2, 3*/),
                ),
            ),
        );
        let c2 = _mm_unpackhi_ps(c, c);
        let d2 = _mm_unpackhi_ps(d, d);
        *p2 = _mm_add_ss(*p2, _mm_mul_ss(a2, c2));
        *p2 = _mm_add_ss(*p2, _mm_mul_ss(b2, d2));
    }
}

// Optimized motor * motor operation
#[inline]
#[allow(clippy::many_single_char_names)]
pub fn gp_mm(a: __m128, b: __m128, c: __m128, d: __m128, e: &mut __m128, f: &mut __m128) {
    unsafe {
        // (a0 c0 - a1 c1 - a2 c2 - a3 c3) +
        // (a0 c1 + a3 c2 + a1 c0 - a2 c3) e23 +
        // (a0 c2 + a1 c3 + a2 c0 - a3 c1) e31 +
        // (a0 c3 + a2 c1 + a3 c0 - a1 c2) e12 +
        //
        // (a0 d0 + b0 c0 + a1 d1 + b1 c1 + a2 d2 + a3 d3 + b2 c2 + b3 c3)
        //  e0123 +
        // (a0 d1 + b1 c0 + a3 d2 + b3 c2 - a1 d0 - a2 d3 - b0 c1 - b2 c3)
        //  e01 +
        // (a0 d2 + b2 c0 + a1 d3 + b1 c3 - a2 d0 - a3 d1 - b0 c2 - b3 c1)
        //  e02 +
        // (a0 d3 + b3 c0 + a2 d1 + b2 c1 - a3 d0 - a1 d2 - b0 c3 - b1 c2)
        //  e03
        let a_xxxx = _mm_shuffle_ps(a, a, 0 /* 0, 0, 0, 0 */);
        let a_zyzw = _mm_shuffle_ps(a, a, 230 /* 3, 2, 1, 2 */);
        let a_ywyz = _mm_shuffle_ps(a, a, 157 /* 2, 1, 3, 1 */);
        let a_wzwy = _mm_shuffle_ps(a, a, 123 /*1, 3, 2, 3 */);
        let c_wwyz = _mm_shuffle_ps(c, c, 159 /*2, 1, 3, 3*/);
        let c_yzwy = _mm_shuffle_ps(c, c, 121 /* 1, 3, 2, 1 */);
        let s_flip = _mm_set_ss(-0.);

        *e = _mm_mul_ps(a_xxxx, c);
        let mut t = _mm_mul_ps(a_ywyz, c_yzwy);
        t = _mm_add_ps(
            t,
            _mm_mul_ps(a_zyzw, _mm_shuffle_ps(c, c, 2 /*0, 0, 0, 2*/)),
        );
        t = _mm_xor_ps(t, s_flip);
        *e = _mm_add_ps(*e, t);
        *e = _mm_sub_ps(*e, _mm_mul_ps(a_wzwy, c_wwyz));

        *f = _mm_mul_ps(a_xxxx, d);
        *f = _mm_add_ps(*f, _mm_mul_ps(b, _mm_shuffle_ps(c, c, 0 /* 0, 0, 0, 0 */)));
        *f = _mm_add_ps(
            *f,
            _mm_mul_ps(a_ywyz, _mm_shuffle_ps(d, d, 121 /* 1, 3, 2, 1 */)),
        );
        *f = _mm_add_ps(
            *f,
            _mm_mul_ps(_mm_shuffle_ps(b, b, 157 /* 2, 1, 3, 1 */), c_yzwy),
        );
        t = _mm_mul_ps(a_zyzw, _mm_shuffle_ps(d, d, 2 /*0, 0, 0, 2*/));
        t = _mm_add_ps(
            t,
            _mm_mul_ps(a_wzwy, _mm_shuffle_ps(d, d, 159 /*2, 1, 3, 3*/)),
        );
        t = _mm_add_ps(
            t,
            _mm_mul_ps(
                _mm_shuffle_ps(b, b, 2 /*0, 0, 0, 2*/),
                _mm_shuffle_ps(c, c, 230 /* 3, 2, 1, 2 */),
            ),
        );
        t = _mm_add_ps(
            t,
            _mm_mul_ps(_mm_shuffle_ps(b, b, 123 /*1, 3, 2, 3*/), c_wwyz),
        );
        t = _mm_xor_ps(t, s_flip);
        *f = _mm_sub_ps(*f, t);
    }
}
