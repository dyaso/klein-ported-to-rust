#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
pub fn mat4x4_12(translated: bool, normalized: bool, b: __m128, c: &__m128, res: &mut [__m128; 4]) {
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

    unsafe {
        let mut buf = <[f32; 4]>::default();
        _mm_storeu_ps(&mut buf[0], _mm_mul_ps(b, b));
        let b0_2 = buf[0];
        let b1_2 = buf[1];
        let b2_2 = buf[2];
        let b3_2 = buf[3];

        // The first column of the matrix we need to produce contains the scale
        // factors of the x-coordinate (a1). This can be read off as:
        //
        // b0^2 + b1^2 - b3^2 - b2^2
        // 2(b1 b2 - b3 b0)
        // 2(b2 b0 + b1 b3)
        // 0
        res[0] = _mm_mul_ps(b, _mm_shuffle_ps(b, b, 8 /* 0, 0, 2, 0 */));
        let mut tmp = _mm_mul_ps(
            _mm_shuffle_ps(b, b, 29 /* 0, 1, 3, 1 */),
            _mm_shuffle_ps(b, b, 49 /* 0, 3, 0, 1 */),
        );
        tmp = _mm_xor_ps(_mm_set_ps(0., 0., -0., 0.), tmp);
        res[0] = _mm_mul_ps(_mm_set_ps(0., 2., 2., 1.), _mm_add_ps(res[0], tmp));
        res[0] = _mm_sub_ps(res[0], _mm_set_ss(b3_2 + b2_2));

        // We can perform the same exercise for y (a2) (the second column):
        //
        // 2(b0 b3 + b2 b1)
        // (-b1^2 - b3^2 + b0^2 + b2^2)
        // 2(b2 b3 - b0 b1)
        // 0
        //	ref __m128 c1 = ref res[1];
        res[1] = _mm_mul_ps(b, _mm_shuffle_ps(b, b, 55 /* 0, 3, 1, 3 */));
        tmp = _mm_mul_ps(
            _mm_shuffle_ps(b, b, 14 /* 0, 0, 3, 2 */),
            _mm_shuffle_ps(b, b, 29 /* 0, 1, 3, 1 */),
        );
        tmp = _mm_xor_ps(_mm_set_ps(0., -0., 0., 0.), tmp);
        res[1] = _mm_mul_ps(_mm_set_ps(0., 2., -1., 2.), _mm_add_ps(res[1], tmp));
        res[1] = _mm_add_ps(res[1], _mm_set_ps(0., 0., b0_2 + b2_2, 0.));

        // z (a3)
        //
        // 2(-b0 b2 + b1 b3)
        // 2(b1 b0 + b2 b3)
        // (-b2^2 + b0^2 + b3^2 - b1^2)
        // 0
        // ref __m128 c2 = ref res[2];
        res[2] = _mm_xor_ps(
            _mm_set_ps(0., -0., 0., -0.),
            _mm_mul_ps(b, _mm_shuffle_ps(b, b, 34 /* 0, 2, 0, 2 */)),
        );
        res[2] = _mm_add_ps(
            res[2],
            _mm_mul_ps(
                _mm_shuffle_ps(b, b, 9 /* 0, 0, 2, 1 */),
                _mm_shuffle_ps(b, b, 15 /* 0, 0, 3, 3 */),
            ),
        );
        res[2] = _mm_mul_ps(res[2], _mm_set_ps(0., 1., 2., 2.));
        res[2] = _mm_add_ps(res[2], _mm_set_ps(0., b3_2 - b1_2, 0., 0.));

        // And finally w (a0)
        //
        // 2(b2 c3 - b0 c1 - b3 c2 - b1 c0)
        // 2(b3 c1 - b1 c3 - b0 c2 - b2 c0)
        // 2(b1 c2 - b2 c1 - b0 c3 - b3 c0)
        // b0^2 + b1^2 + b2^2 + b3^2
        //ref __m128 c3 = ref res[3];
        if translated {
            res[3] = _mm_mul_ps(b, _mm_shuffle_ps(*c, *c, 29 /* 0, 1, 3, 1 */));
            res[3] = _mm_add_ps(
                res[3],
                _mm_mul_ps(
                    _mm_shuffle_ps(b, b, 3 /* 0, 0, 0, 3 */),
                    _mm_shuffle_ps(*c, *c, 58 /* 0, 3, 2, 2 */),
                ),
            );
            res[3] = _mm_add_ps(
                res[3],
                _mm_mul_ps(
                    _mm_shuffle_ps(b, b, 57 /* 0, 3, 2, 1 */),
                    _mm_shuffle_ps(*c, *c, 0 /* 0, 0, 0, 0 */),
                ),
            );
            tmp = _mm_mul_ps(
                _mm_shuffle_ps(b, b, 30 /* 0, 1, 3, 2 */),
                _mm_shuffle_ps(*c, *c, 39 /* 0, 2, 1, 3 */),
            );
            res[3] = _mm_mul_ps(_mm_set_ps(0., 2., 2., 2.), _mm_sub_ps(tmp, res[3]));
        }

        #[allow(clippy::collapsible_if)]
        if normalized {
            if is_x86_feature_detected!("sse4.1") {
                res[3] = _mm_blend_ps(res[3], _mm_set_ps(1., 0., 0., 0.), 0b1000);
            } else {
                res[3] = _mm_add_ps(res[3], _mm_set_ps(1., 0., 0., 0.));
            }
        } else {
            if is_x86_feature_detected!("sse4.1") {
                res[3] = _mm_blend_ps(
                    res[3],
                    _mm_set_ps(b0_2 + b1_2 + b2_2 + b3_2, 0., 0., 0.),
                    0b1000
                );
            } else {
                res[3] = _mm_add_ps(res[3], _mm_set_ps(b0_2 + b1_2 + b2_2 + b3_2, 0., 0., 0.));
            }
        }
    }
}
