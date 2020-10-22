#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{dp_bc, hi_dp, hi_dp_bc, rcp_nr1, rsqrt_nr1};

use std::ops::Neg;

/// The `branch` both a line through the origin and also the principal branch of
/// the logarithm of a rotor.
///
/// The rotor branch will be most commonly constructed by taking the
/// logarithm of a normalized rotor. The branch may then be linearly scaled
/// to adjust the "strength" of the rotor, and subsequently re-exponentiated
/// to create the adjusted rotor.
///
/// !!! example
///
/// Suppose we have a rotor $r$ and we wish to produce a rotor
/// $\sqrt[4]{r}$ which performs a quarter of the rotation produced by
/// $r$. We can construct it like so:
///
/// ```c++
///     kln::branch b = r.log();
///     kln::rotor r_4 = (0.25f * b).exp();
/// ```
///
/// !!! note
///
/// The branch of a rotor is technically a `line`, but because there are
/// no translational components, the branch is given its own type for
/// efficiency.

#[derive(Copy, Clone, Debug)]
pub struct Branch {
    pub p1_: __m128,
}

common_operations!(Branch, p1_);

impl Branch {

    /// Construct the branch as the following multivector:
    ///
    /// $$a \mathbf{e}_{23} + b\mathbf{e}_{31} + c\mathbf{e}_{12}$$
    ///
    /// To convince yourself this is a line through the origin, remember that
    /// such a line can be generated using the geometric product of two planes
    /// through the origin.
    pub fn new(a: f32, b: f32, c: f32) -> Branch {
        unsafe {
            Branch {
                p1_: _mm_set_ps(c, b, a, 0.),
            }
        }
    }

    /// If a line is constructed as the regressive product (join) of
    /// two points, the squared norm provided here is the squared
    /// distance between the two points (provided the points are
    /// normalized). Returns $d^2 + e^2 + f^2$.
    pub fn squared_norm(self) -> f32 {
        let mut out: f32 = 0.;
        let dp: __m128 = hi_dp(self.p1_, self.p1_);
        unsafe {
            _mm_store_ss(&mut out, dp);
        }
        out
    }

    /// Returns the square root of the quantity produced by `squared_norm`.
    pub fn norm(self) -> f32 {
        f32::sqrt(self.squared_norm())
    }

    pub fn normalize(&mut self) {
        unsafe {
            let inv_norm: __m128;

            inv_norm = rsqrt_nr1(hi_dp_bc(self.p1_, self.p1_));

            self.p1_ = _mm_mul_ps(self.p1_, inv_norm);
        }
    }

    pub fn invert(&mut self) {
        let inv_norm: __m128 = rsqrt_nr1(hi_dp_bc(self.p1_, self.p1_));
        unsafe {
            self.p1_ = _mm_mul_ps(self.p1_, inv_norm);
            self.p1_ = _mm_mul_ps(self.p1_, inv_norm);
            self.p1_ = _mm_xor_ps(_mm_set_ps(-0., -0., -0., 0.), self.p1_);
        }
    }

    pub fn inverse(self) -> Branch {
        let mut out = Branch::from(self.p1_);
        out.invert();
        out
    }
}




impl Mul<Branch> for f32 {
    type Output = Branch;
    #[inline]
    fn mul(self, p: Branch) -> Branch {
        p * self
    }
}

impl Mul<Branch> for f64 {
    type Output = Branch;
    #[inline]
    fn mul(self, p: Branch) -> Branch {
        p * self as f32
    }
}

impl Mul<Branch> for i32 {
    type Output = Branch;
    #[inline]
    fn mul(self, p: Branch) -> Branch {
        p * self as f32
    }
}

/// Ideal line uniform inverse scale
impl<T: Into<f32>> Div<T> for Branch {
    type Output = Branch;
    #[inline]
    fn div(self, s: T) -> Self {
        unsafe { Branch::from(_mm_mul_ps(self.p1_, rcp_nr1(_mm_set1_ps(s.into())))) }
    }
}

impl Branch {
    /// Store m128 contents into an array of 4 floats
    pub fn store(self, out: &mut f32) {
        unsafe {
            _mm_store_ps(&mut *out, self.p1_);
        }
    }

    get_basis_blade_fn!(e12, e21, p1_, 3);
    get_basis_blade_fn!(e31, e13, p1_, 2);
    get_basis_blade_fn!(e23, e32, p1_, 1);

    pub fn z(self) -> f32 {
        self.e12()
    }

    pub fn y(self) -> f32 {
        self.e31()
    }

    pub fn x(self) -> f32 {
        self.e23()
    }
}

impl Neg for Branch {
    type Output = Self;
    /// Unary minus
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from(unsafe { _mm_xor_ps(self.p1_, _mm_set1_ps(-0.)) })
    }
}
