#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{hi_dp_bc, rcp_nr1, rsqrt_nr1}; //hi_dp, hi_dp_ss

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

// a := p1
// b := p2
// a + b is a general bivector but it is most likely *non-simple* meaning
// that it is neither purely real nor purely ideal.
// Exponentiates the bivector and returns the motor defined by partitions 1
// and 2.

#[inline]
pub fn simd_exp(a: __m128, b: __m128, p1_out: &mut __m128, p2_out: &mut __m128) {
    // The exponential map produces a continuous group of rotations about an
    // axis. We'd *like* to evaluate the exp(a + b) as exp(a)exp(b) but we
    // cannot do that in general because a and b do not commute (consider
    // the differences between the Taylor expansion of exp(ab) and
    // exp(a)exp(b)).

    // First, we need to decompose the bivector into the sum of two
    // commutative bivectors (the product of these two parts will be a
    // scalar multiple of the pseudoscalar; see "Bivector times its ideal
    // axis and vice versa in demo.klein"). To do this, we compute the
    // squared norm of the bivector:
    //
    // NOTE: a sign flip is introduced since the square of a Euclidean
    // line is negative
    //
    // (a1^2 + a2^2 + a3^2) - 2(a1 b1 + a2 b2 + a3 b3) e0123

    unsafe {
        // Check if the bivector we're exponentiating is ideal
        let mask = _mm_movemask_ps(_mm_cmpeq_ps(a, _mm_setzero_ps()));

        if mask == 0xf {
            // When exponentiating an ideal line, the terms past the linear
            // term in the Taylor series expansion vanishes
            *p1_out = _mm_set_ss(1.);
            *p2_out = b;
            return
        }

        // Broadcast dot(a, a) ignoring the scalar component to all components
        // of a2
        let a2 = hi_dp_bc(a, a);
        let ab = hi_dp_bc(a, b);

        // Next, we need the sqrt of that quantity. Since e0123 squares to 0,
        // this has a closed form solution.
        //
        // sqrt(a1^2 + a2^2 + a3^2)
        //  - (a1 b1 + a2 b2 + a3 b3) / sqrt(a1^2 + a2^2 + a3^2) e0123
        //
        // (relabeling) = u + vI
        //
        // (square the above quantity yourself to quickly verify the claim)
        // Maximum relative error < 1.5*2e-12

        let a2_sqrt_rcp = rsqrt_nr1(a2);
        let u = _mm_mul_ps(a2, a2_sqrt_rcp);
        // Don't forget the minus later!
        let minus_v = _mm_mul_ps(ab, a2_sqrt_rcp);

        // Last, we need the reciprocal of the norm to compute the normalized
        // bivector.
        //
        // 1 / sqrt(a1^2 + a2^2 + a3^2)
        //   + (a1 b1 + a2 b2 + a3 b3) / (a1^2 + a2^2 + a3^2)^(3/2) e0123
        //
        // The original bivector * the inverse norm gives us a normalized
        // bivector.
        let norm_real = _mm_mul_ps(a, a2_sqrt_rcp);
        let mut norm_ideal = _mm_mul_ps(b, a2_sqrt_rcp);
        // The real part of the bivector also interacts with the pseudoscalar to
        // produce a portion of the normalized ideal part
        // e12 e0123 = -e03, e31 e0123 = -e02, e23 e0123 = -e01
        // Notice how the products above actually commute
        norm_ideal = _mm_sub_ps(
            norm_ideal,
            _mm_mul_ps(a, _mm_mul_ps(ab, _mm_mul_ps(a2_sqrt_rcp, rcp_nr1(a2)))),
        );

        // The norm * our normalized bivector is the original bivector (a + b).
        // Thus, we have:
        //
        // (u + vI)n = u n + v n e0123
        //
        // Note that n and n e0123 are perpendicular (n e0123 lies on the ideal
        // plane, and all ideal components of n are extinguished after
        // polarization). As a result, we can now decompose the exponential.
        //
        // e^(u n + v n e0123) = e^(u n) e^(v n e0123) =
        // (cosu + sinu n) * (1 + v n e0123) =
        // cosu + sinu n + v n cosu e0123 + v sinu n^2 e0123 =
        // cosu + sinu n + v n cosu e0123 - v sinu e0123
        //
        // where we've used the fact that n is normalized and squares to -1.
        let mut uv = <[f32; 2]>::default();
        _mm_store_ss(&mut uv[0], u);
        // Note the v here corresponds to minus_v
        _mm_store_ss(&mut uv[1], minus_v);

        let mut sincosu = <[f32; 2]>::default();
        sincosu[0] = f32::sin(uv[0]);
        sincosu[1] = f32::cos(uv[0]);

        let sinu = _mm_set1_ps(sincosu[0]);

        *p1_out = _mm_add_ps(
            _mm_set_ps(0., 0., 0., sincosu[1]),
            _mm_mul_ps(sinu, norm_real)
        );

        // The second partition has contributions from both the real and ideal
        // parts.
        let cosu = _mm_set_ps(sincosu[1], sincosu[1], sincosu[1], 0.);
        let minus_vcosu = _mm_mul_ps(minus_v, cosu);
        *p2_out = _mm_mul_ps(sinu, norm_ideal);
        *p2_out = _mm_add_ps(*p2_out, _mm_mul_ps(minus_vcosu, norm_real));
        let minus_vsinu = uv[1] * sincosu[0];
        *p2_out = _mm_add_ps(_mm_set_ps(0., 0., 0., minus_vsinu), *p2_out);
    }
}

#[allow(clippy::many_single_char_names)]
#[inline]
pub fn simd_log(p1: __m128, p2: __m128, p1_out: &mut __m128, p2_out: &mut __m128) {
    // The logarithm follows from the derivation of the exponential. Working
    // backwards, we ended up computing the exponential like so:
    //
    // cosu + sinu n + v n cosu e0123 - v sinu e0123 =
    // (cosu - v sinu e0123) + (sinu + v cosu e0123) n
    //
    // where n is the normalized bivector. If we compute the norm, that will
    // allow us to match it to sinu + vcosu e0123, which will then allow us
    // to deduce u and v.

    // The first thing we need to do is extract only the bivector components
    // from the motor.
    unsafe {
        let bv_mask = _mm_set_ps(1., 1., 1., 0.);
        let a = _mm_mul_ps(bv_mask, p1);
//        let b = _mm_mul_ps(bv_mask, p2);


        // Early out if we're taking the log of a motor without any rotation
        let mask = _mm_movemask_ps(_mm_cmpeq_ps(a, _mm_setzero_ps()));

            println!("set rotation");
        if mask == 0xf {
            println!("no rotation");
            *p1_out = _mm_setzero_ps();
            *p2_out = p2;
            return;
        }

        let b = _mm_mul_ps(bv_mask, p2);








        // Next, we need to compute the norm as in the exponential.
        let a2 = hi_dp_bc(a, a);
        let ab = hi_dp_bc(a, b);
        let a2_sqrt_rcp = rsqrt_nr1(a2);
        let s = _mm_mul_ps(a2, a2_sqrt_rcp);
        let minus_t = _mm_mul_ps(ab, a2_sqrt_rcp);
        // s + t e0123 is the norm of our bivector.

        // Store the scalar component
        let mut p: f32 = 0.;
        _mm_store_ss(&mut p, p1);

        // Store the pseudoscalar component
        let mut q: f32 = 0.;
        _mm_store_ss(&mut q, p2);

        let mut s_scalar: f32 = 0.;
        _mm_store_ss(&mut s_scalar, s);
        let mut t_scalar: f32 = 0.;
        _mm_store_ss(&mut t_scalar, minus_t);
        t_scalar *= -1.;
        // p = cosu
        // q = -v sinu
        // s_scalar = sinu
        // t_scalar = v cosu

        let u: f32;
        let v: f32;
        let p_zero: bool = f32::abs(p) < 1e-6;
        if p_zero {
            u = f32::atan2(-q, t_scalar);
            v = -q / s_scalar;
        } else {
            u = f32::atan2(s_scalar, p);
            v = t_scalar / p;
        }
        // float u = p_zero ? std::atan2(-q, t_scalar) : std::atan2(s_scalar, p);
        // float v = p_zero ? -q / s_scalar : t_scalar / p;

        // Now, (u + v e0123) * n when exponentiated will give us the motor, so
        // (u + v e0123) * n is the logarithm. To proceed, we need to compute
        // the normalized bivector.
        let norm_real = _mm_mul_ps(a, a2_sqrt_rcp);
        let mut norm_ideal = _mm_mul_ps(b, a2_sqrt_rcp);
        norm_ideal = _mm_sub_ps(
            norm_ideal,
            _mm_mul_ps(a, _mm_mul_ps(ab, _mm_mul_ps(a2_sqrt_rcp, rcp_nr1(a2)))),
        );

        let uvec = _mm_set1_ps(u);
        *p1_out = _mm_mul_ps(uvec, norm_real);
        *p2_out = _mm_mul_ps(uvec, norm_ideal);
        *p2_out = _mm_sub_ps(*p2_out, _mm_mul_ps(_mm_set1_ps(v), norm_real));
    }
}
