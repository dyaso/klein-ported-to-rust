// https://github.com/Ziriax/KleinSharp/blob/master/KleinSharp/Source/Detail/x86_matrix.cs

/// Purpose: Provide conversion routines from rotors, motors, and translators to
/// matrices
///
/// Notes:
/// The preferred layout is a column-major layout as mat-mat and mat-vec
/// multiplication is more naturally implemented when defined this way.

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

// Convert a motor to a column-major 4x4
#[inline]
pub fn mat4x4_12(translated: bool, normalized: bool,b: __m128 , __m128* c, __m128* res)
{
	// The derivation of this conversion follows directly from the general
	// expansion of conjugating a point with a motor. See sw312 in
	// klein_sw.hpp for details.
	//
	// LSB
	// (2a0(b2 c3 - b0 c1 - b3 c2 - b1 c0) +
	//  2a3(b1 b3 - b0 b2) +
	//  2a2(b0 b3 + b2 b1) +
	//  a1(b0^2 + b1^2 - b3^2 - b2^2)) e032 // x-coordinate
	//
	// (2a0(b3 c1 - b0 c2 - b1 c3 - b2 c0) +
	//  2a1(b2 b1 - b0 b3) +
	//  2a3(b0 b1 + b3 b2) +
	//  a2(b0^2 + b2^2 - b1^2 - b3^2)) e013 + // y-coordinate
	//
	// (2a0(b1 c2 - b0 c3 - b2 c1 - b3 c0) +
	//  2a2(b3 b2 - b0 b1) +
	//  2a1(b0 b2 + b1 b3) +
	//  a3(b0^2 + b3^2 - b2^2 - b1^2)) e021 + // z-coordinate
	//
	// a0(b0^2 + b1^2 + b2^2 + b3^2) e123 // w-coordinate
	// MSB

	// Store a number of scalar temporaries needed later
	var buf = _mm_mul_ps(b, b);
	float b0_2 = buf.GetElement(0);
	float b1_2 = buf.GetElement(1);
	float b2_2 = buf.GetElement(2);
	float b3_2 = buf.GetElement(3);

	// The first column of the matrix we need to produce contains the scale
	// factors of the x-coordinate (a1). This can be read off as:
	//
	// b0^2 + b1^2 - b3^2 - b2^2
	// 2(b1 b2 - b3 b0)
	// 2(b2 b0 + b1 b3)
	// 0
	ref __m128 c0 = ref res[0];
	c0 = _mm_mul_ps(b, _mm_swizzle_ps(b, 8 /* 0, 0, 2, 0 */));
	__m128 tmp
		 = _mm_mul_ps(_mm_swizzle_ps(b, 29 /* 0, 1, 3, 1 */), _mm_swizzle_ps(b, 49 /* 0, 3, 0, 1 */));
	tmp = _mm_xor_ps(_mm_set_ps(0f, 0f, -0f, 0f), tmp);
	c0 = _mm_mul_ps(_mm_set_ps(0f, 2f, 2f, 1f), _mm_add_ps(c0, tmp));
	c0 = _mm_sub_ps(c0, _mm_set_ss(b3_2 + b2_2));

	// We can perform the same exercise for y (a2) (the second column):
	//
	// 2(b0 b3 + b2 b1)
	// (-b1^2 - b3^2 + b0^2 + b2^2)
	// 2(b2 b3 - b0 b1)
	// 0
	ref __m128 c1 = ref res[1];
	c1 = _mm_mul_ps(b, _mm_swizzle_ps(b, 55 /* 0, 3, 1, 3 */));
	tmp = _mm_mul_ps(_mm_swizzle_ps(b, 14 /* 0, 0, 3, 2 */), _mm_swizzle_ps(b, 29 /* 0, 1, 3, 1 */));
	tmp = _mm_xor_ps(_mm_set_ps(0f, -0f, 0f, 0f), tmp);
	c1 = _mm_mul_ps(_mm_set_ps(0f, 2f, -1f, 2f), _mm_add_ps(c1, tmp));
	c1 = _mm_add_ps(c1, _mm_set_ps(0f, 0f, b0_2 + b2_2, 0f));

	// z (a3)
	//
	// 2(-b0 b2 + b1 b3)
	// 2(b1 b0 + b2 b3)
	// (-b2^2 + b0^2 + b3^2 - b1^2)
	// 0
	ref __m128 c2 = ref res[2];
	c2 = _mm_xor_ps(_mm_set_ps(0f, -0f, 0f, -0f),
						 _mm_mul_ps(b, _mm_swizzle_ps(b, 34 /* 0, 2, 0, 2 */)));
	c2 = _mm_add_ps(
		 c2, _mm_mul_ps(_mm_swizzle_ps(b, 9 /* 0, 0, 2, 1 */), _mm_swizzle_ps(b, 15 /* 0, 0, 3, 3 */)));
	c2 = _mm_mul_ps(c2, _mm_set_ps(0f, 1f, 2f, 2f));
	c2 = _mm_add_ps(c2, _mm_set_ps(0f, b3_2 - b1_2, 0f, 0f));

	// And finally w (a0)
	//
	// 2(b2 c3 - b0 c1 - b3 c2 - b1 c0)
	// 2(b3 c1 - b1 c3 - b0 c2 - b2 c0)
	// 2(b1 c2 - b2 c1 - b0 c3 - b3 c0)
	// b0^2 + b1^2 + b2^2 + b3^2
	ref __m128 c3 = ref res[3];
	if (translated)
	{
		c3 = _mm_mul_ps(b, _mm_swizzle_ps(*c, 29 /* 0, 1, 3, 1 */));
		c3 = _mm_add_ps(
			 c3,
			 _mm_mul_ps(_mm_swizzle_ps(b, 3 /* 0, 0, 0, 3 */), _mm_swizzle_ps(*c, 58 /* 0, 3, 2, 2 */)));
		c3 = _mm_add_ps(
			 c3,
			 _mm_mul_ps(_mm_swizzle_ps(b, 57 /* 0, 3, 2, 1 */), _mm_swizzle_ps(*c, 0 /* 0, 0, 0, 0 */)));
		tmp = _mm_mul_ps(_mm_swizzle_ps(b, 30 /* 0, 1, 3, 2 */), _mm_swizzle_ps(*c, 39 /* 0, 2, 1, 3 */));
		c3 = _mm_mul_ps(_mm_set_ps(0f, 2f, 2f, 2f), _mm_sub_ps(tmp, c3));
	}
	if (normalized)
	{
		c3 = Sse41.IsSupported
			? _mm_blend_ps(c3, _mm_set_ps(1f, 0f, 0f, 0f), 0b1000)
			: _mm_add_ps(c3, _mm_set_ps(1f, 0f, 0f, 0f));
	}
	else
	{
		c3 = Sse41.IsSupported
		? _mm_blend_ps(c3, _mm_set_ps(b0_2 + b1_2 + b2_2 + b3_2, 0f, 0f, 0f), 0b1000)
		: _mm_add_ps(c3, _mm_set_ps(b0_2 + b1_2 + b2_2 + b3_2, 0f, 0f, 0f));
	}
}
}
}