// https://github.com/Ziriax/KleinSharp/blob/master/KleinSharp/Source/Detail/x86_sandwich.cs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
/// p1: (1, e23, e31, e12)
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

#[inline] pub fn sw10(a:__m128, b:__m128, p1:&mut __m128, p2:&mut __m128)		{
	//                       b0(a1^2 + a2^2 + a3^2) +
	// (2a3(a1 b1 + a2 b2) + b3(a3^2 - a1^2 - a2^2)) e12 +
	// (2a1(a2 b2 + a3 b3) + b1(a1^2 - a2^2 - a3^2)) e23 +
	// (2a2(a3 b3 + a1 b1) + b2(a2^2 - a3^2 - a1^2)) e31 +
	//
	// 2a0(a1 b2 - a2 b1) e03
	// 2a0(a2 b3 - a3 b2) e01 +
	// 2a0(a3 b1 - a1 b3) e02 +

    unsafe {
    	let a_zyzw :__m128 = _mm_shuffle_ps(a,a, 230 /* 3, 2, 1, 2 */);
    	let a_ywyz :__m128 = _mm_shuffle_ps(a,a, 157 /* 2, 1, 3, 1 */);
    	let a_wzwy :__m128 = _mm_shuffle_ps(a,a, 123 /* 1, 3, 2, 3 */);

    	let b_xzwy :__m128 = _mm_shuffle_ps(b,b, 120 /* 1, 3, 2, 0 */);

    	let two_zero :__m128 = _mm_set_ps(2., 2., 2., 0.);
    	*p1 = _mm_mul_ps(a, b);
    	*p1 = _mm_add_ps(*p1, _mm_mul_ps(a_wzwy, b_xzwy));
    	*p1 = _mm_mul_ps(*p1, _mm_mul_ps(a_ywyz, two_zero));

    	let mut tmp:__m128 = _mm_mul_ps(a_zyzw, a_zyzw);
    	tmp = _mm_add_ps(tmp, _mm_mul_ps(a_wzwy, a_wzwy));
    	tmp = _mm_xor_ps(tmp, _mm_set_ss(-0.));
    	tmp = _mm_sub_ps(_mm_mul_ps(a_ywyz, a_ywyz), tmp);
    	tmp = _mm_mul_ps(_mm_shuffle_ps(b,b, 156 /* 2, 1, 3, 0 */), tmp);

        let um = _mm_add_ps(*p1, tmp);
    	*p1 = _mm_shuffle_ps(um,um, 120 /* 1, 3, 2, 0 */);

    	*p2 = _mm_mul_ps(a_zyzw, b_xzwy);
    	*p2 = _mm_sub_ps(*p2, _mm_mul_ps(a_wzwy, b));
    	*p2 = _mm_mul_ps(*p2, _mm_mul_ps(_mm_shuffle_ps(a,a, 0 /* 0, 0, 0, 0 */), two_zero));
    	*p2 = _mm_shuffle_ps(*p2,*p2, 120 /* 1, 3, 2, 0 */);
    }
}

#[inline]
pub fn sw20(a:__m128 ,b: __m128) ->__m128		{
			//                       -b0(a1^2 + a2^2 + a3^2) e0123 +
			// (-2a3(a1 b1 + a2 b2) + b3(a1^2 + a2^2 - a3^2)) e03
			// (-2a1(a2 b2 + a3 b3) + b1(a2^2 + a3^2 - a1^2)) e01 +
			// (-2a2(a3 b3 + a1 b1) + b2(a3^2 + a1^2 - a2^2)) e02 +

    unsafe {
		let a_zzwy :__m128 = _mm_shuffle_ps(a,a, 122 /* 1, 3, 2, 2 */);
		let a_wwyz :__m128 = _mm_shuffle_ps(a,a, 159 /* 2, 1, 3, 3 */);

		let mut p2 = _mm_mul_ps(a, b);
		p2 = _mm_add_ps(p2, _mm_mul_ps(a_zzwy, _mm_shuffle_ps(b,b, 120 /* 1, 3, 2, 0 */)));
		p2 = _mm_mul_ps(
			p2, _mm_mul_ps(a_wwyz, _mm_set_ps(-2., -2., -2., 0.)));

		let a_yyzw :__m128 = _mm_shuffle_ps(a,a, 229 /* 3, 2, 1, 1 */);
		let mut tmp :__m128 = _mm_mul_ps(a_yyzw, a_yyzw);
		tmp = _mm_xor_ps(
			_mm_set_ss(-0.), _mm_add_ps(tmp, _mm_mul_ps(a_zzwy, a_zzwy)));
		tmp = _mm_sub_ps(tmp, _mm_mul_ps(a_wwyz, a_wwyz));
		p2 = _mm_add_ps(p2, _mm_mul_ps(tmp, _mm_shuffle_ps(b,b, 156 /* 2, 1, 3, 0 */)));
		p2 = _mm_shuffle_ps(p2,p2, 120 /* 1, 3, 2, 0 */);

		return p2
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

// 		// Apply a translator to a plane.
// 		// Assumes e0123 component of p2 is exactly 0
// 		// p0: (e0, e1, e2, e3)
// 		// p2: (e0123, e01, e02, e03)
// 		// b * a * ~b
// 		// The low component of p2 is expected to be the scalar component instead
// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static __m128 sw02(__m128 a, __m128 b)
// 		{
// 			// (a0 b0^2 + 2a1 b0 b1 + 2a2 b0 b2 + 2a3 b0 b3) e0 +
// 			// (a1 b0^2) e1 +
// 			// (a2 b0^2) e2 +
// 			// (a3 b0^2) e3
// 			//
// 			// Because the plane is projectively equivalent on multiplication by a
// 			// scalar, we can divide the result through by b0^2
// 			//
// 			// (a0 + 2a1 b1 / b0 + 2a2 b2 / b0 + 2a3 b3 / b0) e0 +
// 			// a1 e1 +
// 			// a2 e2 +
// 			// a3 e3
// 			//
// 			// The additive term clearly contains a dot product between the plane's
// 			// normal and the translation axis, demonstrating that the plane
// 			// "doesn't care" about translations along its span. More precisely, the
// 			// plane translates by the projection of the translator on the plane's
// 			// normal.

// 			// a1*b1 + a2*b2 + a3*b3 stored in the low component of tmp
// 			__m128 tmp = hi_dp(a, b);

// 			__m128 inv_b = rcp_nr1(b);
// 			// 2 / b0
// 			inv_b = _mm_add_ss(inv_b, inv_b);
// 			inv_b = _mm_and_ps(inv_b, _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1)));
// 			tmp = _mm_mul_ss(tmp, inv_b);

// 			// Add to the plane
// 			return _mm_add_ps(a, tmp);
// 		}

// 		// Apply a translator to a line
// 		// a := p1 input
// 		// d := p2 input
// 		// c := p2 translator
// 		// out points to the start address of a line (p1, p2)
// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static (__m128, __m128) swL2(__m128 a, __m128 d, __m128 c)
// 		{
// 			// a0 +
// 			// a1 e23 +
// 			// a2 e31 +
// 			// a3 e12 +
// 			//
// 			// (2a0 c0 + d0) e0123 +
// 			// (2(a2 c3 - a3 c2 - a1 c0) + d1) e01 +
// 			// (2(a3 c1 - a1 c3 - a2 c0) + d2) e02 +
// 			// (2(a1 c2 - a2 c1 - a3 c0) + d3) e03

// 			var p1_out = a;

// 			var p2_out = _mm_mul_ps(_mm_swizzle_ps(a, 120 /* 1, 3, 2, 0 */), _mm_swizzle_ps(c, 156 /* 2, 1, 3, 0 */));

// 			// Add and subtract the same quantity in the low component to produce a
// 			// cancellation
// 			p2_out = _mm_sub_ps(
// 				p2_out,
// 				_mm_mul_ps(_mm_swizzle_ps(a, 156 /* 2, 1, 3, 0 */), _mm_swizzle_ps(c, 120 /* 1, 3, 2, 0 */)));
// 			p2_out = _mm_sub_ps(p2_out,
// 				_mm_xor_ps(_mm_mul_ps(a, _mm_swizzle_ps(c, 0 /* 0, 0, 0, 0 */)),
// 					_mm_set_ss(-0f)));
// 			p2_out = _mm_add_ps(p2_out, p2_out);
// 			p2_out = _mm_add_ps(p2_out, d);

// 			return (p1_out, p2_out);
// 		}

// 		// Apply a translator to a point.
// 		// Assumes e0123 component of p2 is exactly 0
// 		// p2: (e0123, e01, e02, e03)
// 		// p3: (e123, e032, e013, e021)
// 		// b * a * ~b
// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static __m128 sw32(__m128 a, __m128 b)
// 		{
// 			// a0 e123 +
// 			// (a1 - 2 a0 b1) e032 +
// 			// (a2 - 2 a0 b2) e013 +
// 			// (a3 - 2 a0 b3) e021

// 			__m128 tmp = _mm_mul_ps(_mm_swizzle_ps(a, 0 /* 0, 0, 0, 0 */), b);
// 			tmp = _mm_mul_ps(_mm_set_ps(-2f, -2f, -2f, 0f), tmp);
// 			tmp = _mm_add_ps(a, tmp);
// 			return tmp;
// 		}

// 		// Apply a motor to a motor (works on lines as well)
// 		// in points to the start of an array of motor inputs (alternating p1 and
// 		// p2) out points to the start of an array of motor outputs (alternating p1
// 		// and p2)
// 		//
// 		// Note: inp and out are permitted to alias iff a == out.
// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		private static (__m128 tmp1, __m128 tmp2, __m128 tmp3) swMMRotation(__m128 b)
// 		{
// 			// p1 block
// 			// a0(b0^2 + b1^2 + b2^2 + b3^2) +
// 			// (a1(b1^2 + b0^2 - b3^2 - b2^2) +
// 			//     2a2(b0 b3 + b1 b2) + 2a3(b1 b3 - b0 b2)) e23 +
// 			// (a2(b2^2 + b0^2 - b1^2 - b3^2) +
// 			//     2a3(b0 b1 + b2 b3) + 2a1(b2 b1 - b0 b3)) e31
// 			// (a3(b3^2 + b0^2 - b2^2 - b1^2) +
// 			//     2a1(b0 b2 + b3 b1) + 2a2(b3 b2 - b0 b1)) e12 +

// 			__m128 b_xwyz = _mm_swizzle_ps(b, 156 /* 2, 1, 3, 0 */);
// 			__m128 b_xzwy = _mm_swizzle_ps(b, 120 /* 1, 3, 2, 0 */);
// 			__m128 b_yxxx = _mm_swizzle_ps(b, 1 /* 0, 0, 0, 1 */);
// 			__m128 b_yxxx_2 = _mm_mul_ps(b_yxxx, b_yxxx);

// 			__m128 tmp1 = _mm_mul_ps(b, b);
// 			tmp1 = _mm_add_ps(tmp1, b_yxxx_2);
// 			__m128 b_tmp = _mm_swizzle_ps(b, 158 /* 2, 1, 3, 2 */);
// 			__m128 tmp2 = _mm_mul_ps(b_tmp, b_tmp);
// 			b_tmp = _mm_swizzle_ps(b, 123 /* 1, 3, 2, 3 */);
// 			tmp2 = _mm_add_ps(tmp2, _mm_mul_ps(b_tmp, b_tmp));
// 			tmp1 = _mm_sub_ps(tmp1, _mm_xor_ps(tmp2, _mm_set_ss(-0f)));
// 			// tmp needs to be scaled by a and set to p1_out

// 			__m128 b_xxxx = _mm_swizzle_ps(b, 0 /* 0, 0, 0, 0 */);
// 			__m128 scale = _mm_set_ps(2f, 2f, 2f, 0f);
// 			tmp2 = _mm_mul_ps(b_xxxx, b_xwyz);
// 			tmp2 = _mm_add_ps(tmp2, _mm_mul_ps(b, b_xzwy));
// 			tmp2 = _mm_mul_ps(tmp2, scale);
// 			// tmp2 needs to be scaled by (a0, a2, a3, a1) and added to p1_out

// 			__m128 tmp3 = _mm_mul_ps(b, b_xwyz);
// 			tmp3 = _mm_sub_ps(tmp3, _mm_mul_ps(b_xxxx, b_xzwy));
// 			tmp3 = _mm_mul_ps(tmp3, scale);
// 			// tmp3 needs to be scaled by (a0, a3, a1, a2) and added to p1_out

// 			// p2 block
// 			// (d coefficients are the components of the input line p2)
// 			// (2a0(b0 c0 - b1 c1 - b2 c2 - b3 c3) +
// 			//  d0(b1^2 + b0^2 + b2^2 + b3^2)) e0123 +
// 			//
// 			// (2a1(b1 c1 - b0 c0 - b3 c3 - b2 c2) +
// 			//  2a3(b1 c3 + b2 c0 + b3 c1 - b0 c2) +
// 			//  2a2(b1 c2 + b0 c3 + b2 c1 - b3 c0) +
// 			//  2d2(b0 b3 + b2 b1) +
// 			//  2d3(b1 b3 - b0 b2) +
// 			//  d1(b0^2 + b1^2 - b3^2 - b2^2)) e01 +
// 			//
// 			// (2a2(b2 c2 - b0 c0 - b3 c3 - b1 c1) +
// 			//  2a1(b2 c1 + b3 c0 + b1 c2 - b0 c3) +
// 			//  2a3(b2 c3 + b0 c1 + b3 c2 - b1 c0) +
// 			//  2d3(b0 b1 + b3 b2) +
// 			//  2d1(b2 b1 - b0 b3) +
// 			//  d2(b0^2 + b2^2 - b1^2 - b3^2)) e02 +
// 			//
// 			// (2a3(b3 c3 - b0 c0 - b1 c1 - b2 c2) +
// 			//  2a2(b3 c2 + b1 c0 + b2 c3 - b0 c1) +
// 			//  2a1(b3 c1 + b0 c2 + b1 c3 - b2 c0) +
// 			//  2d1(b0 b2 + b1 b3) +
// 			//  2d2(b3 b2 - b0 b1) +
// 			//  d3(b0^2 + b3^2 - b2^2 - b1^2)) e03

// 			// Rotation

// 			// tmp scaled by d and added to p2
// 			// tmp2 scaled by (d0, d2, d3, d1) and added to p2
// 			// tmp3 scaled by (d0, d3, d1, d2) and added to p2

// 			return (tmp1, tmp2, tmp3);
// 		}

// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		private static (__m128 tmp4, __m128 tmp5, __m128 tmp6) swMMTranslation(__m128 b, __m128 c)
// 		{
// 			__m128 b_xwyz = _mm_swizzle_ps(b, 156 /* 2, 1, 3, 0 */);
// 			__m128 b_xzwy = _mm_swizzle_ps(b, 120 /* 1, 3, 2, 0 */);
// 			__m128 b_yxxx = _mm_swizzle_ps(b, 1 /* 0, 0, 0, 1 */);
// 			__m128 b_xxxx = _mm_swizzle_ps(b, 0 /* 0, 0, 0, 0 */);
// 			__m128 scale = _mm_set_ps(2f, 2f, 2f, 0f);

// 			// Translation
// 			__m128 czero = _mm_swizzle_ps(c, 0 /* 0, 0, 0, 0 */);
// 			__m128 c_xzwy = _mm_swizzle_ps(c, 120 /* 1, 3, 2, 0 */);
// 			__m128 c_xwyz = _mm_swizzle_ps(c, 156 /* 2, 1, 3, 0 */);

// 			var tmp4 = _mm_mul_ps(b, c);
// 			tmp4 = _mm_sub_ps(
// 				tmp4, _mm_mul_ps(b_yxxx, _mm_swizzle_ps(c, 1 /* 0, 0, 0, 1 */)));
// 			tmp4 = _mm_sub_ps(tmp4,
// 				_mm_mul_ps(_mm_swizzle_ps(b, 126 /* 1, 3, 3, 2 */),
// 					_mm_swizzle_ps(c, 126 /* 1, 3, 3, 2 */)));
// 			tmp4 = _mm_sub_ps(tmp4,
// 				_mm_mul_ps(_mm_swizzle_ps(b, 155 /* 2, 1, 2, 3 */),
// 					_mm_swizzle_ps(c, 155 /* 2, 1, 2, 3 */)));
// 			tmp4 = _mm_add_ps(tmp4, tmp4);

// 			var tmp5 = _mm_mul_ps(b, c_xwyz);
// 			tmp5 = _mm_add_ps(tmp5, _mm_mul_ps(b_xzwy, czero));
// 			tmp5 = _mm_add_ps(tmp5, _mm_mul_ps(b_xwyz, c));
// 			tmp5 = _mm_sub_ps(tmp5, _mm_mul_ps(b_xxxx, c_xzwy));
// 			tmp5 = _mm_mul_ps(tmp5, scale);

// 			var tmp6 = _mm_mul_ps(b, c_xzwy);
// 			tmp6 = _mm_add_ps(tmp6, _mm_mul_ps(b_xxxx, c_xwyz));
// 			tmp6 = _mm_add_ps(tmp6, _mm_mul_ps(b_xzwy, c));
// 			tmp6 = _mm_sub_ps(tmp6, _mm_mul_ps(b_xwyz, czero));
// 			tmp6 = _mm_mul_ps(tmp6, scale);

// 			return (tmp4, tmp5, tmp6);
// 		}

// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static unsafe void swMM(
// 			bool Translate, bool InputP2,
// 			__m128* inp, __m128 b, __m128 c,
// 			__m128* res, int count)
// 		{
// 			var (tmp1, tmp2, tmp3) = swMMRotation(b);

// 			var (tmp4, tmp5, tmp6) = Translate ? swMMTranslation(b, c) : (__m128.Zero, __m128.Zero, __m128.Zero);

// 			int stride = InputP2 ? 2 : 1;
// 			for (int i = 0; i < count; ++i)
// 			{
// 				ref __m128 p1_in = ref inp[stride * i]; // a
// 				__m128 p1_in_xzwy = _mm_swizzle_ps(p1_in, 120 /* 1, 3, 2, 0 */);
// 				__m128 p1_in_xwyz = _mm_swizzle_ps(p1_in, 156 /* 2, 1, 3, 0 */);

// 				ref __m128 p1_out = ref res[stride * i];

// 				p1_out = _mm_mul_ps(tmp1, p1_in);
// 				p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp2, p1_in_xzwy));
// 				p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp3, p1_in_xwyz));

// 				if (InputP2)
// 				{
// 					ref __m128 p2_in = ref inp[2 * i + 1]; // d
// 					ref __m128 p2_out = ref res[2 * i + 1];
// 					p2_out = _mm_mul_ps(tmp1, p2_in);
// 					p2_out = _mm_add_ps(
// 						p2_out, _mm_mul_ps(tmp2, _mm_swizzle_ps(p2_in, 120 /* 1, 3, 2, 0 */)));
// 					p2_out = _mm_add_ps(
// 						p2_out, _mm_mul_ps(tmp3, _mm_swizzle_ps(p2_in, 156 /* 2, 1, 3, 0 */)));
// 				}

// 				// If what is being applied is a rotor, the non-directional
// 				// components of the line are left untouched
// 				if (Translate)
// 				{
// 					ref __m128 p2_out = ref res[2 * i + 1];
// 					p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp4, p1_in));
// 					p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp5, p1_in_xwyz));
// 					p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp6, p1_in_xzwy));
// 				}
// 			}
// 		}

// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static (__m128, __m128) swMM(
// 			__m128 inp1, __m128 inp2,
// 			__m128 b, __m128 c)
// 		{
// 			var (tmp1, tmp2, tmp3) = swMMRotation(b);
// 			var (tmp4, tmp5, tmp6) = swMMTranslation(b, c);

// 			__m128 p1_in_xzwy = _mm_swizzle_ps(inp1, 120 /* 1, 3, 2, 0 */);
// 			__m128 p1_in_xwyz = _mm_swizzle_ps(inp1, 156 /* 2, 1, 3, 0 */);

// 			var p1_out = _mm_mul_ps(tmp1, inp1);
// 			p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp2, p1_in_xzwy));
// 			p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp3, p1_in_xwyz));

// 			var p2_out = _mm_mul_ps(tmp1, inp2);
// 			p2_out = _mm_add_ps(
// 				p2_out, _mm_mul_ps(tmp2, _mm_swizzle_ps(inp2, 120 /* 1, 3, 2, 0 */)));
// 			p2_out = _mm_add_ps(
// 				p2_out, _mm_mul_ps(tmp3, _mm_swizzle_ps(inp2, 156 /* 2, 1, 3, 0 */)));

// 			// Translate
// 			p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp4, inp1));
// 			p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp5, p1_in_xwyz));
// 			p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp6, p1_in_xzwy));

// 			return (p1_out, p2_out);
// 		}

// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static (__m128, __m128) swMM(__m128 inp1, __m128 inp2, __m128 b)
// 		{
// 			var (tmp1, tmp2, tmp3) = swMMRotation(b);

// 			__m128 p1_in_xzwy = _mm_swizzle_ps(inp1, 120 /* 1, 3, 2, 0 */);
// 			__m128 p1_in_xwyz = _mm_swizzle_ps(inp1, 156 /* 2, 1, 3, 0 */);

// 			var p1_out = _mm_mul_ps(tmp1, inp1);
// 			p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp2, p1_in_xzwy));
// 			p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp3, p1_in_xwyz));

// 			var p2_out = _mm_mul_ps(tmp1, inp2);
// 			p2_out = _mm_add_ps(
// 				p2_out, _mm_mul_ps(tmp2, _mm_swizzle_ps(inp2, 120 /* 1, 3, 2, 0 */)));
// 			p2_out = _mm_add_ps(
// 				p2_out, _mm_mul_ps(tmp3, _mm_swizzle_ps(inp2, 156 /* 2, 1, 3, 0 */)));

// 			return (p1_out, p2_out);
// 		}

// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static __m128 swMM(__m128 inp1, __m128 b)
// 		{
// 			var (tmp1, tmp2, tmp3) = swMMRotation(b);

// 			__m128 p1_in_xzwy = _mm_swizzle_ps(inp1, 120 /* 1, 3, 2, 0 */);
// 			__m128 p1_in_xwyz = _mm_swizzle_ps(inp1, 156 /* 2, 1, 3, 0 */);

// 			var p1_out = _mm_mul_ps(tmp1, inp1);
// 			p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp2, p1_in_xzwy));
// 			p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp3, p1_in_xwyz));
// 			return p1_out;
// 		}

// 		// Apply a motor to a plane
// 		// a := p0
// 		// b := p1
// 		// c := p2
// 		// If Translate is false, c is ignored (rotor application).
// 		// If Variadic is true, a and out must point to a contiguous block of memory
// 		// equivalent to __m128[count]
// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		private static (__m128, __m128, __m128, __m128) sw012Common(bool Translate, __m128 b, __m128 c)
// 		{
// 			// LSB
// 			//
// 			// (2a3(b0 c3 + b1 c2 + b3 c0 - b2 c1) +
// 			//  2a2(b0 c2 + b3 c1 + b2 c0 - b1 c3) +
// 			//  2a1(b0 c1 + b2 c3 + b1 c0 - b3 c2) +
// 			//  a0 (b2^2 + b1^2 + b0^2 + b3^2)) e0 +
// 			//
// 			// (2a2(b0 b3 + b2 b1) +
// 			//  2a3(b1 b3 - b0 b2) +
// 			//  a1 (b0^2 + b1^2 - b3^2 - b2^2)) e1 +
// 			//
// 			// (2a3(b0 b1 + b3 b2) +
// 			//  2a1(b2 b1 - b0 b3) +
// 			//  a2 (b0^2 + b2^2 - b1^2 - b3^2)) e2 +
// 			//
// 			// (2a1(b0 b2 + b1 b3) +
// 			//  2a2(b3 b2 - b0 b1) +
// 			//  a3 (b0^2 + b3^2 - b2^2 - b1^2)) e3
// 			//
// 			// MSB
// 			//
// 			// Note the similarity between the results here and the rotor and
// 			// translator applied to the plane. The e1, e2, and e3 components do not
// 			// participate in the translation and are identical to the result after
// 			// the rotor was applied to the plane. The e0 component is displaced
// 			// similarly to the manner in which it is displaced after application of
// 			// a translator.

// 			// Double-cover scale
// 			__m128 dc_scale = _mm_set_ps(2f, 2f, 2f, 1f);
// 			__m128 b_xwyz = _mm_swizzle_ps(b, 156 /* 2, 1, 3, 0 */);
// 			__m128 b_xzwy = _mm_swizzle_ps(b, 120 /* 1, 3, 2, 0 */);
// 			__m128 b_xxxx = _mm_swizzle_ps(b, 0 /* 0, 0, 0, 0 */);

// 			__m128 tmp1
// 				= _mm_mul_ps(_mm_swizzle_ps(b, 2 /* 0, 0, 0, 2 */), _mm_swizzle_ps(b, 158 /* 2, 1, 3, 2 */));
// 			tmp1 = _mm_add_ps(
// 				tmp1,
// 				_mm_mul_ps(_mm_swizzle_ps(b, 121 /* 1, 3, 2, 1 */), _mm_swizzle_ps(b, 229 /* 3, 2, 1, 1 */)));
// 			// Scale later with (a0, a2, a3, a1)
// 			tmp1 = _mm_mul_ps(tmp1, dc_scale);

// 			__m128 tmp2 = _mm_mul_ps(b, b_xwyz);

// 			tmp2 = _mm_sub_ps(tmp2,
// 				_mm_xor_ps(_mm_set_ss(-0f),
// 					_mm_mul_ps(_mm_swizzle_ps(b, 3 /* 0, 0, 0, 3 */),
// 						_mm_swizzle_ps(b, 123 /* 1, 3, 2, 3 */))));
// 			// Scale later with (a0, a3, a1, a2)
// 			tmp2 = _mm_mul_ps(tmp2, dc_scale);

// 			// Alternately add and subtract to improve low component stability
// 			__m128 tmp3 = _mm_mul_ps(b, b);
// 			tmp3 = _mm_sub_ps(tmp3, _mm_mul_ps(b_xwyz, b_xwyz));
// 			tmp3 = _mm_add_ps(tmp3, _mm_mul_ps(b_xxxx, b_xxxx));
// 			tmp3 = _mm_sub_ps(tmp3, _mm_mul_ps(b_xzwy, b_xzwy));
// 			// Scale later with a

// 			// Compute
// 			// 0 * _ +
// 			// 2a1(b0 c1 + b2 c3 + b1 c0 - b3 c2) +
// 			// 2a2(b0 c2 + b3 c1 + b2 c0 - b1 c3) +
// 			// 2a3(b0 c3 + b1 c2 + b3 c0 - b2 c1)
// 			// by decomposing into four vectors, factoring out the a components

// 			__m128 tmp4 = default;
// 			if (Translate)
// 			{
// 				tmp4 = _mm_mul_ps(b_xxxx, c);
// 				tmp4 = _mm_add_ps(
// 					tmp4, _mm_mul_ps(b_xzwy, _mm_swizzle_ps(c, 156 /* 2, 1, 3, 0 */)));
// 				tmp4 = _mm_add_ps(tmp4, _mm_mul_ps(b, _mm_swizzle_ps(c, 0 /* 0, 0, 0, 0 */)));

// 				// NOTE: The high component of tmp4 is meaningless here
// 				tmp4 = _mm_sub_ps(
// 					tmp4, _mm_mul_ps(b_xwyz, _mm_swizzle_ps(c, 120 /* 1, 3, 2, 0 */)));
// 				tmp4 = _mm_mul_ps(tmp4, dc_scale);
// 			}

// 			return (tmp1, tmp2, tmp3, tmp4);
// 		}

// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static unsafe void sw012(bool Translate, __m128* a, __m128 b, __m128 c, __m128* res, int count)
// 		{
// 			var (tmp1, tmp2, tmp3, tmp4) = sw012Common(Translate, b, c);

// 			// The temporaries (tmp1, tmp2, tmp3, tmp4) strictly only have a
// 			// dependence on b and c.
// 			for (int i = 0; i < count; ++i)
// 			{
// 				// Compute the lower block for components e1, e2, and e3
// 				ref __m128 p = ref res[i];
// 				p = _mm_mul_ps(tmp1, _mm_swizzle_ps(a[i], 120 /* 1, 3, 2, 0 */));
// 				p = _mm_add_ps(p, _mm_mul_ps(tmp2, _mm_swizzle_ps(a[i], 156 /* 2, 1, 3, 0 */)));
// 				p = _mm_add_ps(p, _mm_mul_ps(tmp3, a[i]));

// 				if (Translate)
// 				{
// 					__m128 tmp5 = hi_dp(tmp4, a[i]);
// 					p = _mm_add_ps(p, tmp5);
// 				}
// 			}
// 		}

// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static __m128 sw012(bool Translate, __m128 a, __m128 b, __m128 c)
// 		{
// 			var (tmp1, tmp2, tmp3, tmp4) = sw012Common(Translate, b, c);

// 			// The temporaries (tmp1, tmp2, tmp3, tmp4) strictly only have a dependence on b and c.
// 			// Compute the lower block for components e1, e2, and e3
// 			var p = _mm_mul_ps(tmp1, _mm_swizzle_ps(a, 120 /* 1, 3, 2, 0 */));
// 			p = _mm_add_ps(p, _mm_mul_ps(tmp2, _mm_swizzle_ps(a, 156 /* 2, 1, 3, 0 */)));
// 			p = _mm_add_ps(p, _mm_mul_ps(tmp3, a));

// 			if (Translate)
// 			{
// 				__m128 tmp5 = hi_dp(tmp4, a);
// 				p = _mm_add_ps(p, tmp5);
// 			}

// 			return p;
// 		}

// 		// Apply a motor to a point
// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static unsafe (__m128, __m128, __m128, __m128) sw312Common(bool Translate, __m128 b, __m128 c)
// 		{
// 			// LSB
// 			// a0(b1^2 + b0^2 + b2^2 + b3^2) e123 +
// 			//
// 			// (2a0(b2 c3 - b0 c1 - b3 c2 - b1 c0) +
// 			//  2a3(b1 b3 - b0 b2) +
// 			//  2a2(b0 b3 +  b2 b1) +
// 			//  a1(b0^2 + b1^2 - b3^2 - b2^2)) e032
// 			//
// 			// (2a0(b3 c1 - b0 c2 - b1 c3 - b2 c0) +
// 			//  2a1(b2 b1 - b0 b3) +
// 			//  2a3(b0 b1 + b3 b2) +
// 			//  a2(b0^2 + b2^2 - b1^2 - b3^2)) e013 +
// 			//
// 			// (2a0(b1 c2 - b0 c3 - b2 c1 - b3 c0) +
// 			//  2a2(b3 b2 - b0 b1) +
// 			//  2a1(b0 b2 + b1 b3) +
// 			//  a3(b0^2 + b3^2 - b2^2 - b1^2)) e021 +
// 			// MSB
// 			//
// 			// Sanity check: For c1 = c2 = c3 = 0, the computation becomes
// 			// indistinguishable from a rotor application and the homogeneous
// 			// coordinate a0 does not participate. As an additional sanity check,
// 			// note that for a normalized rotor and homogenous point, the e123
// 			// component will remain unity.

// 			__m128 two = _mm_set_ps(2f, 2f, 2f, 0f);
// 			__m128 b_xxxx = _mm_swizzle_ps(b, 0 /* 0, 0, 0, 0 */);
// 			__m128 b_xwyz = _mm_swizzle_ps(b, 156 /* 2, 1, 3, 0 */);
// 			__m128 b_xzwy = _mm_swizzle_ps(b, 120 /* 1, 3, 2, 0 */);

// 			__m128 tmp1 = _mm_mul_ps(b, b_xwyz);
// 			tmp1 = _mm_sub_ps(tmp1, _mm_mul_ps(b_xxxx, b_xzwy));
// 			tmp1 = _mm_mul_ps(tmp1, two);
// 			// tmp1 needs to be scaled by (_, a3, a1, a2)

// 			__m128 tmp2 = _mm_mul_ps(b_xxxx, b_xwyz);
// 			tmp2 = _mm_add_ps(tmp2, _mm_mul_ps(b_xzwy, b));
// 			tmp2 = _mm_mul_ps(tmp2, two);
// 			// tmp2 needs to be scaled by (_, a2, a3, a1)

// 			__m128 tmp3 = _mm_mul_ps(b, b);
// 			__m128 b_tmp = _mm_swizzle_ps(b, 1 /* 0, 0, 0, 1 */);
// 			tmp3 = _mm_add_ps(tmp3, _mm_mul_ps(b_tmp, b_tmp));
// 			b_tmp = _mm_swizzle_ps(b, 158 /* 2, 1, 3, 2 */);
// 			__m128 tmp4 = _mm_mul_ps(b_tmp, b_tmp);
// 			b_tmp = _mm_swizzle_ps(b, 123 /* 1, 3, 2, 3 */);
// 			tmp4 = _mm_add_ps(tmp4, _mm_mul_ps(b_tmp, b_tmp));
// 			tmp3 = _mm_sub_ps(tmp3, _mm_xor_ps(tmp4, _mm_set_ss(-0f)));
// 			// tmp3 needs to be scaled by (a0, a1, a2, a3)

// 			if (Translate)
// 			{
// 				tmp4 = _mm_mul_ps(b_xzwy, _mm_swizzle_ps(c, 156 /* 2, 1, 3, 0 */));
// 				tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b_xxxx, c));
// 				tmp4 = _mm_sub_ps(
// 					 tmp4, _mm_mul_ps(b_xwyz, _mm_swizzle_ps(c, 120 /* 1, 3, 2, 0 */)));
// 				tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b, _mm_swizzle_ps(c, 0 /* 0, 0, 0, 0 */)));

// 				// Mask low component and scale other components by 2
// 				tmp4 = _mm_mul_ps(tmp4, two);
// 				// tmp4 needs to be scaled by (_, a0, a0, a0)
// 			}

// 			return (tmp1, tmp2, tmp3, tmp4);
// 		}

// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static unsafe void sw312(bool Translate, __m128* a, __m128 b, __m128 c, __m128* res, int count)
// 		{
// 			var (tmp1, tmp2, tmp3, tmp4) = sw312Common(Translate, b, c);

// 			for (int i = 0; i < count; ++i)
// 			{
// 				ref __m128 p = ref res[i];
// 				p = _mm_mul_ps(tmp1, _mm_swizzle_ps(a[i], 156 /* 2, 1, 3, 0 */));
// 				p = _mm_add_ps(p, _mm_mul_ps(tmp2, _mm_swizzle_ps(a[i], 120 /* 1, 3, 2, 0 */)));
// 				p = _mm_add_ps(p, _mm_mul_ps(tmp3, a[i]));

// 				if (Translate)
// 				{
// 					p = _mm_add_ps(
// 						 p, _mm_mul_ps(tmp4, _mm_swizzle_ps(a[i], 0 /* 0, 0, 0, 0 */)));
// 				}
// 			}
// 		}

// 		// Apply a motor to one point
// 		[MethodImpl(MethodImplOptions.AggressiveInlining)]
// 		public static __m128 sw312(bool Translate, __m128 a, __m128 b, __m128 c)
// 		{
// 			var (tmp1, tmp2, tmp3, tmp4) = sw312Common(Translate, b, c);

// 			var p = _mm_mul_ps(tmp1, _mm_swizzle_ps(a, 156 /* 2, 1, 3, 0 */));
// 			p = _mm_add_ps(p, _mm_mul_ps(tmp2, _mm_swizzle_ps(a, 120 /* 1, 3, 2, 0 */)));
// 			p = _mm_add_ps(p, _mm_mul_ps(tmp3, a));

// 			if (Translate)
// 			{
// 				p = _mm_add_ps(p, _mm_mul_ps(tmp4, _mm_swizzle_ps(a, 0 /* 0, 0, 0, 0 */)));
// 			}

// 			return p;
// 		}

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
