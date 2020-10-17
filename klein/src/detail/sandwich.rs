// https://github.com/Ziriax/KleinSharp/blob/master/KleinSharp/Source/Detail/x86_sandwich.cs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{hi_dp, hi_dp_ss, rcp_nr1}; //, hi_dp, hi_dp_bc, rsqrt_nr1};

/// File: sandwich.hpp
/// Purpose: Define functions of the form swAB where A and B are partition
/// indices. Each function so-defined computes the sandwich operator using vector
/// intrinsics. The partition index determines which basis elements are present
/// in each XMM component of the operand.
///
/// Notes:
/// 1. The first argument is always the TARGET which is the multivector to apply
///    the sandwich operator to.
/// 2. The second operator MAY be a bivector or motor (sandwiching with
///    a point or vector isn't supported at this time).
/// 3. For efficiency, the sandwich operator is NOT implemented in terms of two
///    geometric products and a reversion. The result is nevertheless equivalent.

/// Partition memory layouts
///     LSB --> MSB
/// p0: (e0, e1, e2, e3)
/// .1: (1, e23, e31, e12)
/// p2: (e0123, e01, e02, e03)
/// p3: (e123, e032, e013, e021)

/// Reflect a plane through another plane
/// b * a * b
#[inline]
pub fn sw00(a: __m128, b: __m128, p0_out: &mut __m128) {
    // (2a0(a2 b2 + a3 b3 + a1 b1) - b0(a1^2 + a2^2 + a3^2)) e0 +
    // (2a1(a2 b2 + a3 b3)         + b1(a1^2 - a2^2 - a3^2)) e1 +
    // (2a2(a3 b3 + a1 b1)         + b2(a2^2 - a3^2 - a1^2)) e2 +
    // (2a3(a1 b1 + a2 b2)         + b3(a3^2 - a1^2 - a2^2)) e3

    unsafe {
        let a_zzwy = _mm_shuffle_ps(a, a, 122 /* 1, 3, 2, 2 */);
        let a_wwyz = _mm_shuffle_ps(a, a, 159 /* 2, 1, 3, 3 */);

        // Left block
        let mut tmp = _mm_mul_ps(a_zzwy, _mm_shuffle_ps(b, b, 122 /* 1, 3, 2, 2 */));
        tmp = _mm_add_ps(
            tmp,
            _mm_mul_ps(a_wwyz, _mm_shuffle_ps(b, b, 159 /* 2, 1, 3, 3 */)),
        );

        let a1 = _mm_movehdup_ps(a);
        let b1 = _mm_movehdup_ps(b);
        tmp = _mm_add_ss(tmp, _mm_mul_ss(a1, b1));
        tmp = _mm_mul_ps(tmp, _mm_add_ps(a, a));

        // Right block
        let a_yyzw = _mm_shuffle_ps(a, a, 229 /* 3, 2, 1, 1 */);
        let mut tmp2 = _mm_xor_ps(_mm_mul_ps(a_yyzw, a_yyzw), _mm_set_ss(-0.0));
        tmp2 = _mm_sub_ps(tmp2, _mm_mul_ps(a_zzwy, a_zzwy));
        tmp2 = _mm_sub_ps(tmp2, _mm_mul_ps(a_wwyz, a_wwyz));
        tmp2 = _mm_mul_ps(tmp2, b);

        *p0_out = _mm_add_ps(tmp, tmp2);
    }
}

#[inline]
pub fn sw10(a: __m128, b: __m128, p1: &mut __m128, p2: &mut __m128) {
    //                       b0(a1^2 + a2^2 + a3^2) +
    // (2a3(a1 b1 + a2 b2) + b3(a3^2 - a1^2 - a2^2)) e12 +
    // (2a1(a2 b2 + a3 b3) + b1(a1^2 - a2^2 - a3^2)) e23 +
    // (2a2(a3 b3 + a1 b1) + b2(a2^2 - a3^2 - a1^2)) e31 +
    //
    // 2a0(a1 b2 - a2 b1) e03
    // 2a0(a2 b3 - a3 b2) e01 +
    // 2a0(a3 b1 - a1 b3) e02 +

    unsafe {
        let a_zyzw: __m128 = _mm_shuffle_ps(a, a, 230 /* 3, 2, 1, 2 */);
        let a_ywyz: __m128 = _mm_shuffle_ps(a, a, 157 /* 2, 1, 3, 1 */);
        let a_wzwy: __m128 = _mm_shuffle_ps(a, a, 123 /* 1, 3, 2, 3 */);

        let b_xzwy: __m128 = _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */);

        let two_zero: __m128 = _mm_set_ps(2., 2., 2., 0.);
        *p1 = _mm_mul_ps(a, b);
        *p1 = _mm_add_ps(*p1, _mm_mul_ps(a_wzwy, b_xzwy));
        *p1 = _mm_mul_ps(*p1, _mm_mul_ps(a_ywyz, two_zero));

        let mut tmp: __m128 = _mm_mul_ps(a_zyzw, a_zyzw);
        tmp = _mm_add_ps(tmp, _mm_mul_ps(a_wzwy, a_wzwy));
        tmp = _mm_xor_ps(tmp, _mm_set_ss(-0.));
        tmp = _mm_sub_ps(_mm_mul_ps(a_ywyz, a_ywyz), tmp);
        tmp = _mm_mul_ps(_mm_shuffle_ps(b, b, 156 /* 2, 1, 3, 0 */), tmp);

        let um = _mm_add_ps(*p1, tmp);
        *p1 = _mm_shuffle_ps(um, um, 120 /* 1, 3, 2, 0 */);

        *p2 = _mm_mul_ps(a_zyzw, b_xzwy);
        *p2 = _mm_sub_ps(*p2, _mm_mul_ps(a_wzwy, b));
        *p2 = _mm_mul_ps(
            *p2,
            _mm_mul_ps(_mm_shuffle_ps(a, a, 0 /* 0, 0, 0, 0 */), two_zero),
        );
        *p2 = _mm_shuffle_ps(*p2, *p2, 120 /* 1, 3, 2, 0 */);
    }
}

#[inline]
pub fn sw20(a: __m128, b: __m128) -> __m128 {
    //                       -b0(a1^2 + a2^2 + a3^2) e0123 +
    // (-2a3(a1 b1 + a2 b2) + b3(a1^2 + a2^2 - a3^2)) e03
    // (-2a1(a2 b2 + a3 b3) + b1(a2^2 + a3^2 - a1^2)) e01 +
    // (-2a2(a3 b3 + a1 b1) + b2(a3^2 + a1^2 - a2^2)) e02 +

    unsafe {
        let a_zzwy: __m128 = _mm_shuffle_ps(a, a, 122 /* 1, 3, 2, 2 */);
        let a_wwyz: __m128 = _mm_shuffle_ps(a, a, 159 /* 2, 1, 3, 3 */);

        let mut p2 = _mm_mul_ps(a, b);
        p2 = _mm_add_ps(
            p2,
            _mm_mul_ps(a_zzwy, _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */)),
        );
        p2 = _mm_mul_ps(p2, _mm_mul_ps(a_wwyz, _mm_set_ps(-2., -2., -2., 0.)));

        let a_yyzw: __m128 = _mm_shuffle_ps(a, a, 229 /* 3, 2, 1, 1 */);
        let mut tmp: __m128 = _mm_mul_ps(a_yyzw, a_yyzw);
        tmp = _mm_xor_ps(_mm_set_ss(-0.), _mm_add_ps(tmp, _mm_mul_ps(a_zzwy, a_zzwy)));
        tmp = _mm_sub_ps(tmp, _mm_mul_ps(a_wwyz, a_wwyz));
        p2 = _mm_add_ps(
            p2,
            _mm_mul_ps(tmp, _mm_shuffle_ps(b, b, 156 /* 2, 1, 3, 0 */)),
        );
        p2 = _mm_shuffle_ps(p2, p2, 120 /* 1, 3, 2, 0 */);

        return p2;
    }
}

#[inline]
pub fn sw30(a: __m128, b: __m128) -> __m128 {
    //                                b0(a1^2 + a2^2 + a3^2)  e123 +
    // (-2a1(a0 b0 + a3 b3 + a2 b2) + b1(a2^2 + a3^2 - a1^2)) e032 +
    // (-2a2(a0 b0 + a1 b1 + a3 b3) + b2(a3^2 + a1^2 - a2^2)) e013 +
    // (-2a3(a0 b0 + a2 b2 + a1 b1) + b3(a1^2 + a2^2 - a3^2)) e021
    unsafe {
        let a_zwyz = _mm_shuffle_ps(a, a, 158 /* 2, 1, 3, 2 */);
        let a_yzwy = _mm_shuffle_ps(a, a, 121 /* 1, 3, 2, 1 */);

        let mut p3_out = _mm_mul_ps(
            _mm_shuffle_ps(a, a, 0 /* 0, 0, 0, 0 */),
            _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */),
        );
        p3_out = _mm_add_ps(
            p3_out,
            _mm_mul_ps(a_zwyz, _mm_shuffle_ps(b, b, 156 /* 2, 1, 3, 0 */)),
        );
        p3_out = _mm_add_ps(
            p3_out,
            _mm_mul_ps(a_yzwy, _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */)),
        );
        p3_out = _mm_mul_ps(p3_out, _mm_mul_ps(a, _mm_set_ps(-2.0, -2.0, -2.0, 0.0)));

        let mut tmp = _mm_mul_ps(a_yzwy, a_yzwy);
        tmp = _mm_add_ps(tmp, _mm_mul_ps(a_zwyz, a_zwyz));
        let a_wyzw = _mm_shuffle_ps(a, a, 231 /* 3, 2, 1, 3 */);
        tmp = _mm_sub_ps(
            tmp,
            _mm_xor_ps(_mm_mul_ps(a_wyzw, a_wyzw), _mm_set_ss(-0.0)),
        );

        p3_out = _mm_add_ps(p3_out, _mm_mul_ps(b, tmp));

        return p3_out;
    }
}

// Apply a translator to a plane.
// Assumes e0123 component of p2 is exactly 0
// p0: (e0, e1, e2, e3)
// p2: (e0123, e01, e02, e03)
// b * a * ~b
// The low component of p2 is expected to be the scalar component instead
#[inline]
pub fn sw02(a: __m128, b: __m128) -> __m128 {
    // (a0 b0^2 + 2a1 b0 b1 + 2a2 b0 b2 + 2a3 b0 b3) e0 +
    // (a1 b0^2) e1 +
    // (a2 b0^2) e2 +
    // (a3 b0^2) e3
    //
    // Because the plane is projectively equivalent on multiplication by a
    // scalar, we can divide the result through by b0^2
    //
    // (a0 + 2a1 b1 / b0 + 2a2 b2 / b0 + 2a3 b3 / b0) e0 +
    // a1 e1 +
    // a2 e2 +
    // a3 e3
    //
    // The additive term clearly contains a dot product between the plane's
    // normal and the translation axis, demonstrating that the plane
    // "doesn't care" about translations along its span. More precisely, the
    // plane translates by the projection of the translator on the plane's
    // normal.

    // a1*b1 + a2*b2 + a3*b3 stored in the low component of tmp
    let mut tmp: __m128 = hi_dp(a, b);

    let mut inv_b: __m128 = rcp_nr1(b);
    unsafe {
        // 2 / b0
        inv_b = _mm_add_ss(inv_b, inv_b);
        inv_b = _mm_and_ps(inv_b, _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1)));
        tmp = _mm_mul_ss(tmp, inv_b);

        // Add to the plane
        return _mm_add_ps(a, tmp);
    }
}

// Apply a translator to a line
// a := p1 input
// d := p2 input
// c := p2 translator
#[inline]
pub fn swL2(a: __m128, d: __m128, c: __m128) -> __m128 {
    // a0 +
    // a1 e23 +
    // a2 e31 +
    // a3 e12 +
    //
    // (2a0 c0 + d0) e0123 +
    // (2(a2 c3 - a3 c2 - a1 c0) + d1) e01 +
    // (2(a3 c1 - a1 c3 - a2 c0) + d2) e02 +
    // (2(a1 c2 - a2 c1 - a3 c0) + d3) e03

    unsafe {
        let mut p2_out = _mm_mul_ps(
            _mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */),
            _mm_shuffle_ps(c, c, 156 /* 2, 1, 3, 0 */),
        );

        // Add and subtract the same quantity in the low component to produce a
        // cancellation
        p2_out = _mm_sub_ps(
            p2_out,
            _mm_mul_ps(
                _mm_shuffle_ps(a, a, 156 /* 2, 1, 3, 0 */),
                _mm_shuffle_ps(c, c, 120 /* 1, 3, 2, 0 */),
            ),
        );
        p2_out = _mm_sub_ps(
            p2_out,
            _mm_xor_ps(
                _mm_mul_ps(a, _mm_shuffle_ps(c, c, 0 /* 0, 0, 0, 0 */)),
                _mm_set_ss(-0.),
            ),
        );
        p2_out = _mm_add_ps(p2_out, p2_out);
        p2_out = _mm_add_ps(p2_out, d);

        return p2_out;
    }
}

// Apply a translator to a point.
// Assumes e0123 component of p2 is exactly 0
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)
// b * a * ~b
#[inline]
pub fn sw32(a: __m128, b: __m128) -> __m128 {
    // a0 e123 +
    // (a1 - 2 a0 b1) e032 +
    // (a2 - 2 a0 b2) e013 +
    // (a3 - 2 a0 b3) e021

    unsafe {
        let mut tmp: __m128 = _mm_mul_ps(_mm_shuffle_ps(a, a, 0 /* 0, 0, 0, 0 */), b);
        tmp = _mm_mul_ps(_mm_set_ps(-2., -2., -2., 0.), tmp);
        tmp = _mm_add_ps(a, tmp);
        return tmp;
    }
}

// Apply a motor to a motor (works on lines as well)
// in points to the start of an array of motor inputs (alternating p1 and
// p2) out points to the start of an array of motor outputs (alternating p1
// and p2)
//
// Note: inp and out are permitted to alias iff a == out.
#[inline]
pub fn swMMRotation(b: __m128) -> (__m128, __m128, __m128) {
    // p1 block
    // a0(b0^2 + b1^2 + b2^2 + b3^2) +
    // (a1(b1^2 + b0^2 - b3^2 - b2^2) +
    //     2a2(b0 b3 + b1 b2) + 2a3(b1 b3 - b0 b2)) e23 +
    // (a2(b2^2 + b0^2 - b1^2 - b3^2) +
    //     2a3(b0 b1 + b2 b3) + 2a1(b2 b1 - b0 b3)) e31
    // (a3(b3^2 + b0^2 - b2^2 - b1^2) +
    //     2a1(b0 b2 + b3 b1) + 2a2(b3 b2 - b0 b1)) e12 +

    unsafe {
        let b_xwyz: __m128 = _mm_shuffle_ps(b, b, 156 /* 2, 1, 3, 0 */);
        let b_xzwy: __m128 = _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */);
        let b_yxxx: __m128 = _mm_shuffle_ps(b, b, 1 /* 0, 0, 0, 1 */);
        let b_yxxx_2: __m128 = _mm_mul_ps(b_yxxx, b_yxxx);

        let mut tmp1 = _mm_mul_ps(b, b);
        tmp1 = _mm_add_ps(tmp1, b_yxxx_2);
        let mut b_tmp = _mm_shuffle_ps(b, b, 158 /* 2, 1, 3, 2 */);
        let mut tmp2 = _mm_mul_ps(b_tmp, b_tmp);
        b_tmp = _mm_shuffle_ps(b, b, 123 /* 1, 3, 2, 3 */);
        tmp2 = _mm_add_ps(tmp2, _mm_mul_ps(b_tmp, b_tmp));
        tmp1 = _mm_sub_ps(tmp1, _mm_xor_ps(tmp2, _mm_set_ss(-0.)));
        // tmp needs to be scaled by a and set to p1_out

        let b_xxxx = _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */);
        let scale = _mm_set_ps(2., 2., 2., 0.);
        tmp2 = _mm_mul_ps(b_xxxx, b_xwyz);
        tmp2 = _mm_add_ps(tmp2, _mm_mul_ps(b, b_xzwy));
        tmp2 = _mm_mul_ps(tmp2, scale);
        // tmp2 needs to be scaled by (a0, a2, a3, a1) and added to p1_out

        let mut tmp3 = _mm_mul_ps(b, b_xwyz);
        tmp3 = _mm_sub_ps(tmp3, _mm_mul_ps(b_xxxx, b_xzwy));
        tmp3 = _mm_mul_ps(tmp3, scale);
        // tmp3 needs to be scaled by (a0, a3, a1, a2) and added to p1_out

        // p2 block
        // (d coefficients are the components of the input line p2)
        // (2a0(b0 c0 - b1 c1 - b2 c2 - b3 c3) +
        //  d0(b1^2 + b0^2 + b2^2 + b3^2)) e0123 +
        //
        // (2a1(b1 c1 - b0 c0 - b3 c3 - b2 c2) +
        //  2a3(b1 c3 + b2 c0 + b3 c1 - b0 c2) +
        //  2a2(b1 c2 + b0 c3 + b2 c1 - b3 c0) +
        //  2d2(b0 b3 + b2 b1) +
        //  2d3(b1 b3 - b0 b2) +
        //  d1(b0^2 + b1^2 - b3^2 - b2^2)) e01 +
        //
        // (2a2(b2 c2 - b0 c0 - b3 c3 - b1 c1) +
        //  2a1(b2 c1 + b3 c0 + b1 c2 - b0 c3) +
        //  2a3(b2 c3 + b0 c1 + b3 c2 - b1 c0) +
        //  2d3(b0 b1 + b3 b2) +
        //  2d1(b2 b1 - b0 b3) +
        //  d2(b0^2 + b2^2 - b1^2 - b3^2)) e02 +
        //
        // (2a3(b3 c3 - b0 c0 - b1 c1 - b2 c2) +
        //  2a2(b3 c2 + b1 c0 + b2 c3 - b0 c1) +
        //  2a1(b3 c1 + b0 c2 + b1 c3 - b2 c0) +
        //  2d1(b0 b2 + b1 b3) +
        //  2d2(b3 b2 - b0 b1) +
        //  d3(b0^2 + b3^2 - b2^2 - b1^2)) e03

        // Rotation

        // tmp scaled by d and added to p2
        // tmp2 scaled by (d0, d2, d3, d1) and added to p2
        // tmp3 scaled by (d0, d3, d1, d2) and added to p2

        return (tmp1, tmp2, tmp3);
    }
}

#[inline]
pub fn swMMTranslation(b: __m128, c: __m128) -> (__m128, __m128, __m128) {
    unsafe {
        let b_xwyz = _mm_shuffle_ps(b, b, 156 /* 2, 1, 3, 0 */);
        let b_xzwy = _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */);
        let b_yxxx = _mm_shuffle_ps(b, b, 1 /* 0, 0, 0, 1 */);
        let b_xxxx = _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */);
        let scale = _mm_set_ps(2., 2., 2., 0.);

        // Translation
        let czero = _mm_shuffle_ps(c, c, 0 /* 0, 0, 0, 0 */);
        let c_xzwy = _mm_shuffle_ps(c, c, 120 /* 1, 3, 2, 0 */);
        let c_xwyz = _mm_shuffle_ps(c, c, 156 /* 2, 1, 3, 0 */);

        let mut tmp4 = _mm_mul_ps(b, c);
        tmp4 = _mm_sub_ps(
            tmp4,
            _mm_mul_ps(b_yxxx, _mm_shuffle_ps(c, c, 1 /* 0, 0, 0, 1 */)),
        );
        tmp4 = _mm_sub_ps(
            tmp4,
            _mm_mul_ps(
                _mm_shuffle_ps(b, b, 126 /* 1, 3, 3, 2 */),
                _mm_shuffle_ps(c, c, 126 /* 1, 3, 3, 2 */),
            ),
        );
        tmp4 = _mm_sub_ps(
            tmp4,
            _mm_mul_ps(
                _mm_shuffle_ps(b, b, 155 /* 2, 1, 2, 3 */),
                _mm_shuffle_ps(c, c, 155 /* 2, 1, 2, 3 */),
            ),
        );
        tmp4 = _mm_add_ps(tmp4, tmp4);

        let mut tmp5 = _mm_mul_ps(b, c_xwyz);
        tmp5 = _mm_add_ps(tmp5, _mm_mul_ps(b_xzwy, czero));
        tmp5 = _mm_add_ps(tmp5, _mm_mul_ps(b_xwyz, c));
        tmp5 = _mm_sub_ps(tmp5, _mm_mul_ps(b_xxxx, c_xzwy));
        tmp5 = _mm_mul_ps(tmp5, scale);

        let mut tmp6 = _mm_mul_ps(b, c_xzwy);
        tmp6 = _mm_add_ps(tmp6, _mm_mul_ps(b_xxxx, c_xwyz));
        tmp6 = _mm_add_ps(tmp6, _mm_mul_ps(b_xzwy, c));
        tmp6 = _mm_sub_ps(tmp6, _mm_mul_ps(b_xwyz, czero));
        tmp6 = _mm_mul_ps(tmp6, scale);

        return (tmp4, tmp5, tmp6);
    }
}

use crate::Line;

#[inline]
pub fn swMM_seven(
	translate: bool, InputP2: bool,
	inp: &[Line], b: __m128, c: __m128,
	res: &mut [Line], count:usize) {
	let (tmp1, tmp2, tmp3) = swMMRotation(b);

    unsafe{

    	let (mut tmp4, mut tmp5, mut tmp6) = (_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps());
        if translate {
            let (ttmp4, ttmp5, ttmp6) = swMMTranslation(b, c);
            tmp4 = ttmp4;
            tmp5 = ttmp5;
            tmp6 = ttmp6;
        }

        let mut stride = 1;
        if InputP2 {
            stride = 2;
        }

    	for i in 0..count {
    		let p1_in = inp[i].p1_; // a
    		let p1_in_xzwy = _mm_shuffle_ps(p1_in,p1_in, 120 /* 1, 3, 2, 0 */);
    		let p1_in_xwyz = _mm_shuffle_ps(p1_in,p1_in, 156 /* 2, 1, 3, 0 */);

    		// ref __m128 p1_out = ref res[stride * i];

    		res[i].p1_ = _mm_mul_ps(tmp1, p1_in);
    		res[i].p1_ = _mm_add_ps(res[i].p1_, _mm_mul_ps(tmp2, p1_in_xzwy));
    		res[i].p1_ = _mm_add_ps(res[i].p1_, _mm_mul_ps(tmp3, p1_in_xwyz));

    		if InputP2 {
    			let p2_in = inp[i].p2_; // d
    			// let res[2*i + 1] = ref res[2 * i + 1];
    			res[i].p2_ = _mm_mul_ps(tmp1, p2_in);
    			res[i].p2_ = _mm_add_ps(
    				res[i].p2_, _mm_mul_ps(tmp2, _mm_shuffle_ps(p2_in,p2_in, 120 /* 1, 3, 2, 0 */)));
    			res[i].p2_ = _mm_add_ps(
    				res[i].p2_, _mm_mul_ps(tmp3, _mm_shuffle_ps(p2_in,p2_in, 156 /* 2, 1, 3, 0 */)));
    		}

    		// If what is being applied is a rotor, the non-directional
    		// components of the line are left untouched
    		if translate
    		{
    			//ref __m128 res[i].p2_ = ref res[2 * i + 1];
    			res[i].p2_ = _mm_add_ps(res[i].p2_, _mm_mul_ps(tmp4, p1_in));
    			res[i].p2_ = _mm_add_ps(res[i].p2_, _mm_mul_ps(tmp5, p1_in_xwyz));
    			res[i].p2_ = _mm_add_ps(res[i].p2_, _mm_mul_ps(tmp6, p1_in_xzwy));
    		}
    	}
    }
}

#[inline]
pub fn swMM_four(inp1: __m128, inp2: __m128, b: __m128, c: __m128) -> (__m128, __m128) {
    unsafe {
        let (tmp1, tmp2, tmp3) = swMMRotation(b);
        let (tmp4, tmp5, tmp6) = swMMTranslation(b, c);

        let p1_in_xzwy = _mm_shuffle_ps(inp1, inp1, 120 /* 1, 3, 2, 0 */);
        let p1_in_xwyz = _mm_shuffle_ps(inp1, inp1, 156 /* 2, 1, 3, 0 */);

        let mut p1_out = _mm_mul_ps(tmp1, inp1);
        p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp2, p1_in_xzwy));
        p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp3, p1_in_xwyz));

        let mut p2_out = _mm_mul_ps(tmp1, inp2);
        p2_out = _mm_add_ps(
            p2_out,
            _mm_mul_ps(tmp2, _mm_shuffle_ps(inp2, inp2, 120 /* 1, 3, 2, 0 */)),
        );
        p2_out = _mm_add_ps(
            p2_out,
            _mm_mul_ps(tmp3, _mm_shuffle_ps(inp2, inp2, 156 /* 2, 1, 3, 0 */)),
        );

        // translate
        p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp4, inp1));
        p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp5, p1_in_xwyz));
        p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp6, p1_in_xzwy));

        return (p1_out, p2_out);
    }
}

#[inline]
pub fn swMM_three(inp1: __m128, inp2: __m128, b: __m128) -> (__m128, __m128) {
    unsafe {
        let (tmp1, tmp2, tmp3) = swMMRotation(b);

        let p1_in_xzwy: __m128 = _mm_shuffle_ps(inp1, inp1, 120 /* 1, 3, 2, 0 */);
        let p1_in_xwyz: __m128 = _mm_shuffle_ps(inp1, inp1, 156 /* 2, 1, 3, 0 */);

        let mut p1_out = _mm_mul_ps(tmp1, inp1);
        p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp2, p1_in_xzwy));
        p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp3, p1_in_xwyz));

        let mut p2_out = _mm_mul_ps(tmp1, inp2);
        p2_out = _mm_add_ps(
            p2_out,
            _mm_mul_ps(tmp2, _mm_shuffle_ps(inp2, inp2, 120 /* 1, 3, 2, 0 */)),
        );
        p2_out = _mm_add_ps(
            p2_out,
            _mm_mul_ps(tmp3, _mm_shuffle_ps(inp2, inp2, 156 /* 2, 1, 3, 0 */)),
        );

        return (p1_out, p2_out);
    }
}

#[inline]
pub fn swMM_two(inp1: __m128, b: __m128) -> __m128{
	let (tmp1, tmp2, tmp3) = swMMRotation(b);

    unsafe{

    	let p1_in_xzwy = _mm_shuffle_ps(inp1, inp1, 120 /* 1, 3, 2, 0 */);
    	let p1_in_xwyz = _mm_shuffle_ps(inp1, inp1, 156 /* 2, 1, 3, 0 */);

    	let mut p1_out = _mm_mul_ps(tmp1, inp1);
    	p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp2, p1_in_xzwy));
    	p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp3, p1_in_xwyz));
    	return p1_out
    }
}

// Apply a motor to a plane
// a := p0
// b := p1
// c := p2
// If translate is false, c is ignored (rotor application).
// If Variadic is true, a and out must point to a contiguous block of memory
// equivalent to __m128[count]
#[inline]
pub fn sw012Common(translate: bool, b: __m128, c: __m128) -> (__m128, __m128, __m128, __m128) {
    // LSB
    //
    // (2a3(b0 c3 + b1 c2 + b3 c0 - b2 c1) +
    //  2a2(b0 c2 + b3 c1 + b2 c0 - b1 c3) +
    //  2a1(b0 c1 + b2 c3 + b1 c0 - b3 c2) +
    //  a0 (b2^2 + b1^2 + b0^2 + b3^2)) e0 +
    //
    // (2a2(b0 b3 + b2 b1) +
    //  2a3(b1 b3 - b0 b2) +
    //  a1 (b0^2 + b1^2 - b3^2 - b2^2)) e1 +
    //
    // (2a3(b0 b1 + b3 b2) +
    //  2a1(b2 b1 - b0 b3) +
    //  a2 (b0^2 + b2^2 - b1^2 - b3^2)) e2 +
    //
    // (2a1(b0 b2 + b1 b3) +
    //  2a2(b3 b2 - b0 b1) +
    //  a3 (b0^2 + b3^2 - b2^2 - b1^2)) e3
    //
    // MSB
    //
    // Note the similarity between the results here and the rotor and
    // translator applied to the plane. The e1, e2, and e3 components do not
    // participate in the translation and are identical to the result after
    // the rotor was applied to the plane. The e0 component is displaced
    // similarly to the manner in which it is displaced after application of
    // a translator.

    unsafe {
        // Double-cover scale
        let dc_scale: __m128 = _mm_set_ps(2., 2., 2., 1.);
        let b_xwyz: __m128 = _mm_shuffle_ps(b, b, 156 /* 2, 1, 3, 0 */);
        let b_xzwy: __m128 = _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */);
        let b_xxxx: __m128 = _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */);

        let mut tmp1: __m128 = _mm_mul_ps(
            _mm_shuffle_ps(b, b, 2 /* 0, 0, 0, 2 */),
            _mm_shuffle_ps(b, b, 158 /* 2, 1, 3, 2 */),
        );
        tmp1 = _mm_add_ps(
            tmp1,
            _mm_mul_ps(
                _mm_shuffle_ps(b, b, 121 /* 1, 3, 2, 1 */),
                _mm_shuffle_ps(b, b, 229 /* 3, 2, 1, 1 */),
            ),
        );
        // Scale later with (a0, a2, a3, a1)
        tmp1 = _mm_mul_ps(tmp1, dc_scale);

        let mut tmp2: __m128 = _mm_mul_ps(b, b_xwyz);

        tmp2 = _mm_sub_ps(
            tmp2,
            _mm_xor_ps(
                _mm_set_ss(-0.),
                _mm_mul_ps(
                    _mm_shuffle_ps(b, b, 3 /* 0, 0, 0, 3 */),
                    _mm_shuffle_ps(b, b, 123 /* 1, 3, 2, 3 */),
                ),
            ),
        );
        // Scale later with (a0, a3, a1, a2)
        tmp2 = _mm_mul_ps(tmp2, dc_scale);

        // Alternately add and subtract to improve low component stability
        let mut tmp3: __m128 = _mm_mul_ps(b, b);
        tmp3 = _mm_sub_ps(tmp3, _mm_mul_ps(b_xwyz, b_xwyz));
        tmp3 = _mm_add_ps(tmp3, _mm_mul_ps(b_xxxx, b_xxxx));
        tmp3 = _mm_sub_ps(tmp3, _mm_mul_ps(b_xzwy, b_xzwy));
        // Scale later with a

        // Compute
        // 0 * _ +
        // 2a1(b0 c1 + b2 c3 + b1 c0 - b3 c2) +
        // 2a2(b0 c2 + b3 c1 + b2 c0 - b1 c3) +
        // 2a3(b0 c3 + b1 c2 + b3 c0 - b2 c1)
        // by decomposing into four vectors, factoring out the a components

        let mut tmp4 = _mm_setzero_ps();
        if translate {
            tmp4 = _mm_mul_ps(b_xxxx, c);
            tmp4 = _mm_add_ps(
                tmp4,
                _mm_mul_ps(b_xzwy, _mm_shuffle_ps(c, c, 156 /* 2, 1, 3, 0 */)),
            );
            tmp4 = _mm_add_ps(
                tmp4,
                _mm_mul_ps(b, _mm_shuffle_ps(c, c, 0 /* 0, 0, 0, 0 */)),
            );

            // NOTE: The high component of tmp4 is meaningless here
            tmp4 = _mm_sub_ps(
                tmp4,
                _mm_mul_ps(b_xwyz, _mm_shuffle_ps(c, c, 120 /* 1, 3, 2, 0 */)),
            );
            tmp4 = _mm_mul_ps(tmp4, dc_scale);
        }

        return (tmp1, tmp2, tmp3, tmp4);
    }
}

use crate::Plane;

#[inline]
pub fn sw012_six(translate:bool, a: &[Plane], b: __m128, c: __m128, res: &mut [Plane], count: usize) {
	let (tmp1, tmp2, tmp3, tmp4) = sw012Common(translate, b, c);

	// The temporaries (tmp1, tmp2, tmp3, tmp4) strictly only have a
	// dependence on b and c.
	for i in 0..count {
		// Compute the lower block for components e1, e2, and e3
	    // let ref p = res[i];
        unsafe {
    		res[i].p0_ = _mm_mul_ps(tmp1, _mm_shuffle_ps(a[i].p0_, a[i].p0_, 120 /* 1, 3, 2, 0 */));
    		res[i].p0_ = _mm_add_ps(res[i].p0_, _mm_mul_ps(tmp2, _mm_shuffle_ps(a[i].p0_,a[i].p0_, 156 /* 2, 1, 3, 0 */)));
    		res[i].p0_ = _mm_add_ps(res[i].p0_, _mm_mul_ps(tmp3, a[i].p0_));

    		if translate {
    			let tmp5 = hi_dp(tmp4, a[i].p0_);
    			res[i].p0_ = _mm_add_ps(res[i].p0_, tmp5);
    		}
        }
	}
}

#[inline]
pub fn sw012(translate: bool, a: __m128, b: __m128, c: __m128) -> __m128 {
    let (tmp1, tmp2, tmp3, tmp4) = sw012Common(translate, b, c);

    unsafe {
        // The temporaries (tmp1, tmp2, tmp3, tmp4) strictly only have a dependence on b and c.
        // Compute the lower block for components e1, e2, and e3
        let mut p = _mm_mul_ps(tmp1, _mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */));
        p = _mm_add_ps(
            p,
            _mm_mul_ps(tmp2, _mm_shuffle_ps(a, a, 156 /* 2, 1, 3, 0 */)),
        );
        p = _mm_add_ps(p, _mm_mul_ps(tmp3, a));

        if translate {
            let tmp5 = hi_dp(tmp4, a);
            p = _mm_add_ps(p, tmp5);
        }

        return p;
    }
}

// Apply a motor to a point
#[inline]
pub fn sw312Common(translate: bool, b: __m128, c: __m128) -> (__m128, __m128, __m128, __m128) {
    // LSB
    // a0(b1^2 + b0^2 + b2^2 + b3^2) e123 +
    //
    // (2a0(b2 c3 - b0 c1 - b3 c2 - b1 c0) +
    //  2a3(b1 b3 - b0 b2) +
    //  2a2(b0 b3 +  b2 b1) +
    //  a1(b0^2 + b1^2 - b3^2 - b2^2)) e032
    //
    // (2a0(b3 c1 - b0 c2 - b1 c3 - b2 c0) +
    //  2a1(b2 b1 - b0 b3) +
    //  2a3(b0 b1 + b3 b2) +
    //  a2(b0^2 + b2^2 - b1^2 - b3^2)) e013 +
    //
    // (2a0(b1 c2 - b0 c3 - b2 c1 - b3 c0) +
    //  2a2(b3 b2 - b0 b1) +
    //  2a1(b0 b2 + b1 b3) +
    //  a3(b0^2 + b3^2 - b2^2 - b1^2)) e021 +
    // MSB
    //
    // Sanity check: For c1 = c2 = c3 = 0, the computation becomes
    // indistinguishable from a rotor application and the homogeneous
    // coordinate a0 does not participate. As an additional sanity check,
    // note that for a normalized rotor and homogenous point, the e123
    // component will remain unity.
    unsafe {
        let two = _mm_set_ps(2., 2., 2., 0.);
        let b_xxxx = _mm_shuffle_ps(b, b, 0 /* 0, 0, 0, 0 */);
        let b_xwyz = _mm_shuffle_ps(b, b, 156 /* 2, 1, 3, 0 */);
        let b_xzwy = _mm_shuffle_ps(b, b, 120 /* 1, 3, 2, 0 */);

        let mut tmp1 = _mm_mul_ps(b, b_xwyz);
        tmp1 = _mm_sub_ps(tmp1, _mm_mul_ps(b_xxxx, b_xzwy));
        tmp1 = _mm_mul_ps(tmp1, two);
        // tmp1 needs to be scaled by (_, a3, a1, a2)

        let mut tmp2 = _mm_mul_ps(b_xxxx, b_xwyz);
        tmp2 = _mm_add_ps(tmp2, _mm_mul_ps(b_xzwy, b));
        tmp2 = _mm_mul_ps(tmp2, two);
        // tmp2 needs to be scaled by (_, a2, a3, a1)

        let mut tmp3 = _mm_mul_ps(b, b);
        let mut b_tmp = _mm_shuffle_ps(b, b, 1 /* 0, 0, 0, 1 */);
        tmp3 = _mm_add_ps(tmp3, _mm_mul_ps(b_tmp, b_tmp));
        b_tmp = _mm_shuffle_ps(b, b, 158 /* 2, 1, 3, 2 */);
        let mut tmp4 = _mm_mul_ps(b_tmp, b_tmp);
        b_tmp = _mm_shuffle_ps(b, b, 123 /* 1, 3, 2, 3 */);
        tmp4 = _mm_add_ps(tmp4, _mm_mul_ps(b_tmp, b_tmp));
        tmp3 = _mm_sub_ps(tmp3, _mm_xor_ps(tmp4, _mm_set_ss(-0.)));
        // tmp3 needs to be scaled by (a0, a1, a2, a3)

        if translate {
            tmp4 = _mm_mul_ps(b_xzwy, _mm_shuffle_ps(c, c, 156 /* 2, 1, 3, 0 */));
            tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b_xxxx, c));
            tmp4 = _mm_sub_ps(
                tmp4,
                _mm_mul_ps(b_xwyz, _mm_shuffle_ps(c, c, 120 /* 1, 3, 2, 0 */)),
            );
            tmp4 = _mm_sub_ps(
                tmp4,
                _mm_mul_ps(b, _mm_shuffle_ps(c, c, 0 /* 0, 0, 0, 0 */)),
            );

            // Mask low component and scale other components by 2
            tmp4 = _mm_mul_ps(tmp4, two);
            // tmp4 needs to be scaled by (_, a0, a0, a0)
        }

        return (tmp1, tmp2, tmp3, tmp4);
    }
}

// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static unsafe void sw312(bool translate, __m128* a, __m128 b, __m128 c, __m128* res, int count)
// 		{
// 			var (tmp1, tmp2, tmp3, tmp4) = sw312Common(translate, b, c);

// 			for (int i = 0; i < count; ++i)
// 			{
// 				ref __m128 p = ref res[i];
// 				p = _mm_mul_ps(tmp1, _mm_swizzle_ps(a[i], 156 /* 2, 1, 3, 0 */));
// 				p = _mm_add_ps(p, _mm_mul_ps(tmp2, _mm_swizzle_ps(a[i], 120 /* 1, 3, 2, 0 */)));
// 				p = _mm_add_ps(p, _mm_mul_ps(tmp3, a[i]));

// 				if translate
// 				{
// 					p = _mm_add_ps(
// 						 p, _mm_mul_ps(tmp4, _mm_swizzle_ps(a[i], 0 /* 0, 0, 0, 0 */)));
// 				}
// 			}
// 		}

// Apply a motor to one point
#[inline]
pub fn sw312_four(translate: bool, a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe {
        let (tmp1, tmp2, tmp3, tmp4) = sw312Common(translate, b, c);

        let mut p = _mm_mul_ps(tmp1, _mm_shuffle_ps(a, a, 156 /* 2, 1, 3, 0 */));
        p = _mm_add_ps(
            p,
            _mm_mul_ps(tmp2, _mm_shuffle_ps(a, a, 120 /* 1, 3, 2, 0 */)),
        );
        p = _mm_add_ps(p, _mm_mul_ps(tmp3, a));

        if translate {
            p = _mm_add_ps(
                p,
                _mm_mul_ps(tmp4, _mm_shuffle_ps(a, a, 0 /* 0, 0, 0, 0 */)),
            );
        }

        return p;
    }
}

// 		// Conjugate origin with motor. Unlike other operations the motor MUST be
// 		// normalized prior to usage b is the rotor component (p1) c is the
// 		// translator component (p2)
// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static __m128 swo12(__m128 b, __m128 c)
// 		{
// 			//  (b0^2 + b1^2 + b2^2 + b3^2) e123 +
// 			// 2(b2 c3 - b1 c0 - b0 c1 - b3 c2) e032 +
// 			// 2(b3 c1 - b2 c0 - b0 c2 - b1 c3) e013 +
// 			// 2(b1 c2 - b3 c0 - b0 c3 - b2 c1) e021

// 			__m128 tmp = _mm_mul_ps(b, _mm_swizzle_ps(c, 0 /* 0, 0, 0, 0 */));
// 			tmp = _mm_add_ps(tmp, _mm_mul_ps(_mm_swizzle_ps(b, 0 /* 0, 0, 0, 0 */), c));
// 			tmp = _mm_add_ps(
// 				 tmp,
// 				 _mm_mul_ps(_mm_swizzle_ps(b, 156 /* 2, 1, 3, 0 */), _mm_swizzle_ps(c, 120 /* 1, 3, 2, 0 */)));
// 			tmp = _mm_sub_ps(
// 				 _mm_mul_ps(_mm_swizzle_ps(b, 120 /* 1, 3, 2, 0 */), _mm_swizzle_ps(c, 156 /* 2, 1, 3, 0 */)),
// 				 tmp);
// 			tmp = _mm_mul_ps(tmp, _mm_set_ps(2f, 2f, 2f, 0f));

// 			// b0^2 + b1^2 + b2^2 + b3^2 assumed to equal 1
// 			// Set the low component to unity
// 			return _mm_add_ps(tmp, _mm_set_ss(1f));
// 		}
// 	}
// }
