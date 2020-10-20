#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{dp_bc, hi_dp, hi_dp_bc, rcp_nr1, rsqrt_nr1};

/// Klein provides three line classes: "line", "branch", and "ideal_line". The
/// line class represents a full six-coordinate bivector. The branch contains
/// three non-degenerate components (aka, a line through the origin). The ideal
/// line represents the line at infinity. When the line is created as a meet
/// of two planes or join of two points (or carefully selected Plücker
/// coordinates), it will be a Euclidean line (factorizable as the meet of two
/// vectors).

/// An ideal line represents a line at infinity and corresponds to the
/// multivector:
///
/// $$a\mathbf{e}_{01} + b\mathbf{e}_{02} + c\mathbf{e}_{03}$$

#[derive(Copy, Clone)]
pub struct IdealLine {
    pub p2_: __m128,
}

use std::ops::Neg;

impl IdealLine {
    pub fn new(a: f32, b: f32, c: f32) -> IdealLine {
        unsafe {
            IdealLine {
                p2_: _mm_set_ps(c, b, a, 0.),
            }
        }
    }

    // pub fn from(xmm: __m128) -> IdealLine {
    //     unsafe { IdealLine { p2_: xmm } }
    // }

    pub fn squared_ideal_norm(self) -> f32 {
        let mut out: f32 = 0.;
        let dp: __m128 = hi_dp(self.p2_, self.p2_);
        unsafe {
            _mm_store_ss(&mut out, dp);
        }
        out
    }

    pub fn ideal_norm(self) -> f32 {
        f32::sqrt(self.squared_ideal_norm())
    }

    pub fn normalize(self) -> Self {
        unimplemented!()
    }
}

common_operations!(IdealLine, p2_);




//scalar_multiply!(IdealLine, unsafe { IdealLine::from(_mm_mul_ps(self.p2_, _mm_set1_ps(s))) });

impl Mul<IdealLine> for f32 {
    type Output = IdealLine;
    #[inline]
    fn mul(self, p: IdealLine) -> IdealLine {
        p * self
    }
}
// impl Mul<f64> for IdealLine {
//     type Output = IdealLine;
//     #[inline]
//     fn mul(self, s: f64) -> Self {
//         unsafe { IdealLine::from(_mm_mul_ps(self.p2_, _mm_set1_ps(s as f32))) }
//     }
// }
impl Mul<IdealLine> for f64 {
    type Output = IdealLine;
    #[inline]
    fn mul(self, p: IdealLine) -> IdealLine {
        p * self as f32
    }
}
// impl Mul<i32> for IdealLine {
//     type Output = IdealLine;
//     #[inline]
//     fn mul(self, s: i32) -> Self {
//         unsafe { IdealLine::from(_mm_mul_ps(self.p2_, _mm_set1_ps(s as f32))) }
//     }
// }
impl Mul<IdealLine> for i32 {
    type Output = IdealLine;
    #[inline]
    fn mul(self, p: IdealLine) -> IdealLine {
        p * self as f32
    }
}

/// Ideal line uniform inverse scale
impl Div<f32> for IdealLine {
    type Output = IdealLine;
    #[inline]
    fn div(self, s: f32) -> Self {
        unsafe { IdealLine::from(_mm_mul_ps(self.p2_, rcp_nr1(_mm_set1_ps(s)))) }
    }
}
impl Div<f64> for IdealLine {
    type Output = IdealLine;
    #[inline]
    fn div(self, s: f64) -> Self {
        unsafe { IdealLine::from(_mm_mul_ps(self.p2_, rcp_nr1(_mm_set1_ps(s as f32)))) }
    }
}
impl Div<i32> for IdealLine {
    type Output = IdealLine;
    #[inline]
    fn div(self, s: i32) -> Self {
        unsafe { IdealLine::from(_mm_mul_ps(self.p2_, rcp_nr1(_mm_set1_ps(s as f32)))) }
    }
}

impl IdealLine {
    get_basis_blade_fn!(e01, e10, p2_, 1);
    get_basis_blade_fn!(e02, e20, p2_, 2);
    get_basis_blade_fn!(e03, e30, p2_, 3);
}

impl Neg for IdealLine {
    type Output = Self;
    /// Unary minus
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from(unsafe { _mm_xor_ps(self.p2_, _mm_set1_ps(-0.)) })
    }
}
