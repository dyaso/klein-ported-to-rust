#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::*; //{dp_bc, hi_dp, hi_dp_bc, rcp_nr1, rsqrt_nr1};

/// Klein provides three line classes: "line", "branch", and "ideal_line". The
/// line class represents a full six-coordinate bivector. The branch contains
/// three non-degenerate components (aka, a line through the origin). The ideal
/// line represents the line at infinity. When the line is created as a meet
/// of two planes or join of two points (or carefully selected Plücker
/// coordinates), it will be a Euclidean line (factorizable as the meet of two
/// vectors).
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{Branch, IdealLine};

// p1: (1, e12, e31, e23)
// p2: (e0123, e01, e02, e03)

/// A general line in $\PGA$ is given as a 6-coordinate bivector with a direct
/// correspondence to Plücker coordinates. All lines can be exponentiated using
/// the `exp` method to generate a motor.

#[derive(Copy, Clone)]
pub struct Line {
    pub p1_: __m128,
    pub p2_: __m128,
}

use std::fmt;

impl fmt::Display for Line {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "\nLine\tScalar\te23\te31\te12\te01\te02\te03\te0123\n\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\n",
            self.scalar(),
            self.e23(),
            self.e31(),
            self.e12(),
            self.e01(),
            self.e02(),
            self.e03(),
            self.e0123()
        )
    }
}

use std::ops::Neg;

impl Line {
    //    pub fn default() -> Line {Line {p1_:_mm_setzero_ps(),p2_:_mm_setzero_ps()}}
    pub fn scalar(self) -> f32 {
        let mut out: f32 = 0.;
        unsafe {
            _mm_store_ss(&mut out, self.p1_);
        }
        out
    }
    pub fn e0123(self) -> f32 {
        let mut out: f32 = 0.;
        unsafe {
            _mm_store_ss(&mut out, self.p2_);
        }
        out
    }

    /// A line is specifed by 6 coordinates which correspond to the line's
    /// [Plücker
    /// coordinates](https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates).
    /// The coordinates specified in this way correspond to the following
    /// multivector:
    ///
    /// $$a\mathbf{e}_{01} + b\mathbf{e}_{02} + c\mathbf{e}_{03} +\
    /// d\mathbf{e}_{23} + e\mathbf{e}_{31} + f\mathbf{e}_{12}$$
    #[allow(clippy::many_single_char_names)]
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Line {
        unsafe {
            Line {
                p1_: _mm_set_ps(f, e, d, 0.),
                p2_: _mm_set_ps(c, b, a, 0.),
            }
        }
    }

    pub fn from_branch(other: Branch) -> Line {
        unsafe {
            Line {
                p1_: other.p1_,
                p2_: _mm_setzero_ps(),
            }
        }
    }

    pub fn from_ideal(other: IdealLine) -> Line {
        unsafe {
            Line {
                p1_: _mm_set_ss(1.),
                p2_: other.p2_,
            }
        }
    }

    pub fn from_branch_and_ideal(branch: Branch, ideal: IdealLine) -> Line {
        Line {
            p1_: branch.p1_,
            p2_: ideal.p2_,
        }
    }

    pub fn from(branch: __m128, ideal: __m128) -> Line {
        Line {
            p1_: branch,
            p2_: ideal,
        }
    }

    pub fn default() -> Line {
        unsafe {
            Line {
                p1_: _mm_set_ss(1.),
                p2_: _mm_setzero_ps(),
            }
        }
    }

    /// Returns the square root of the quantity produced by `squared_norm`.
    pub fn norm(self) -> f32 {
        f32::sqrt(self.squared_norm())
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

    /// Normalize a line such that $\ell^2 = -1$.
    pub fn normalize(&mut self) {
        // l = b + c where b is p1 and c is p2
        // l * ~l = |b|^2 - 2(b1 c1 + b2 c2 + b3 c3)e0123
        //
        // sqrt(l*~l) = |b| - (b1 c1 + b2 c2 + b3 c3)/|b| e0123
        //
        // 1/sqrt(l*~l) = 1/|b| + (b1 c1 + b2 c2 + b3 c3)/|b|^3 e0123
        //              = s + t e0123
        unsafe {
            let b2: __m128 = hi_dp_bc(self.p1_, self.p1_);
            let s: __m128 = rsqrt_nr1(b2);
            let bc: __m128 = hi_dp_bc(self.p1_, self.p2_);
            let t: __m128 = _mm_mul_ps(_mm_mul_ps(bc, rcp_nr1(b2)), s);

            // p1 * (s + t e0123) = s * p1 - t p1_perp
            let tmp: __m128 = _mm_mul_ps(self.p2_, s);
            self.p2_ = _mm_sub_ps(tmp, _mm_mul_ps(self.p1_, t));
            self.p1_ = _mm_mul_ps(self.p1_, s);
        }
    }

    pub fn normalized(self) -> Self {
        let mut out = Self::from(self.p1_, self.p2_);
        out.normalize();
        out
    }

    pub fn invert(&mut self) {
        unsafe {
            // s, t computed as in the normalization
            let b2: __m128 = hi_dp_bc(self.p1_, self.p1_);
            let s: __m128 = rsqrt_nr1(b2);
            let bc: __m128 = hi_dp_bc(self.p1_, self.p2_);
            let b2_inv: __m128 = rcp_nr1(b2);
            let t: __m128 = _mm_mul_ps(_mm_mul_ps(bc, b2_inv), s);
            let neg: __m128 = _mm_set_ps(-0., -0., -0., 0.);

            // p1 * (s + t e0123)^2 = (s * p1 - t p1_perp) * (s + t e0123)
            // = s^2 p1 - s t p1_perp - s t p1_perp
            // = s^2 p1 - 2 s t p1_perp
            // p2 * (s + t e0123)^2 = s^2 p2
            // NOTE: s^2 = b2_inv
            let mut st: __m128 = _mm_mul_ps(s, t);
            st = _mm_mul_ps(self.p1_, st);
            self.p2_ = _mm_sub_ps(_mm_mul_ps(self.p2_, b2_inv), _mm_add_ps(st, st));
            self.p2_ = _mm_xor_ps(self.p2_, neg);

            self.p1_ = _mm_xor_ps(_mm_mul_ps(self.p1_, b2_inv), neg);
        }
    }

    pub fn inverse(self) -> Line {
        let mut out = Line::from(self.p1_, self.p2_);
        out.invert();
        out
    }

    pub fn approx_eq(self, other: Line, epsilon: f64) -> bool {
        unsafe {
            let eps: __m128 = _mm_set1_ps(epsilon as f32);
            let neg: __m128 = _mm_set1_ps(-0.);
            let cmp1: __m128 =
                _mm_cmplt_ps(_mm_andnot_ps(neg, _mm_sub_ps(self.p1_, other.p1_)), eps);
            let cmp2: __m128 =
                _mm_cmplt_ps(_mm_andnot_ps(neg, _mm_sub_ps(self.p2_, other.p2_)), eps);
            let cmp: __m128 = _mm_and_ps(cmp1, cmp2);
            _mm_movemask_ps(cmp) == 0xf
        }
    }
}

impl PartialEq for Line {
    fn eq(&self, other: &Line) -> bool {
        unsafe {
            let p1_eq: __m128 = _mm_cmpeq_ps(self.p1_, other.p1_);
            let p2_eq: __m128 = _mm_cmpeq_ps(self.p2_, other.p2_);
            let eq: __m128 = _mm_and_ps(p1_eq, p2_eq);
            _mm_movemask_ps(eq) == 0xf
        }
    }
}

/// Line addition
impl AddAssign for Line {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        unsafe {
            self.p1_ = _mm_add_ps(self.p1_, rhs.p1_);
            self.p2_ = _mm_add_ps(self.p2_, rhs.p2_);
        }
    }
}

/// Line subtraction
impl SubAssign for Line {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        unsafe {
            self.p1_ = _mm_sub_ps(self.p1_, rhs.p1_);
            self.p2_ = _mm_sub_ps(self.p2_, rhs.p2_);
        }
    }
}

/// Line uniform scale
impl<T: Into<f32>> MulAssign<T> for Line {
    #[inline]
    fn mul_assign(&mut self, s: T) {
        unsafe {
            let vs: __m128 = _mm_set1_ps(s.into());
            self.p1_ = _mm_mul_ps(self.p1_, vs);
            self.p2_ = _mm_mul_ps(self.p2_, vs);
        }
    }
}

/// Line uniform inverse scale
impl<T: Into<f32>> DivAssign<T> for Line {
    #[inline]
    fn div_assign(&mut self, s: T) {
        unsafe {
            let vs: __m128 = rcp_nr1(_mm_set1_ps(s.into()));
            self.p1_ = _mm_mul_ps(self.p1_, vs);
            self.p2_ = _mm_mul_ps(self.p2_, vs);
        }
    }
}

impl Line {
    get_basis_blade_fn!(e12, e21, p1_, 3);
    get_basis_blade_fn!(e31, e13, p1_, 2);
    get_basis_blade_fn!(e23, e32, p1_, 1);
    get_basis_blade_fn!(e01, e10, p2_, 1);
    get_basis_blade_fn!(e02, e20, p2_, 2);
    get_basis_blade_fn!(e03, e30, p2_, 3);
}

/// Line addition
impl Add for Line {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Line::from(_mm_add_ps(self.p1_, rhs.p1_), _mm_add_ps(self.p2_, rhs.p2_)) }
    }
}

/// Line subtraction
impl Sub for Line {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Line::from(_mm_sub_ps(self.p1_, rhs.p1_), _mm_sub_ps(self.p2_, rhs.p2_)) }
    }
}

/// Line uniform scale
impl<T: Into<f32>> Mul<T> for Line {
    type Output = Line;
    #[inline]
    fn mul(self, s: T) -> Self {
        unsafe {
            let vs: __m128 = _mm_set1_ps(s.into());
            Line::from(_mm_mul_ps(self.p1_, vs), _mm_mul_ps(self.p2_, vs))
        }
    }
}

// i couldn't work out how to get something like the above to work here
macro_rules! mul_scalar_by_line {
    ($s:ty) => {
        impl Mul<Line> for $s {
            type Output = Line;
            #[inline]
            fn mul(self, l: Line) -> Line {
                l * (self as f32)
            }
        }
    };
}

mul_scalar_by_line!(f32);
mul_scalar_by_line!(f64);
mul_scalar_by_line!(i32);

/// Line uniform inverse scale
impl Div<f32> for Line {
    type Output = Line;
    #[inline]
    fn div(self, s: f32) -> Self {
        unsafe {
            let vs: __m128 = rcp_nr1(_mm_set1_ps(s));
            Line::from(_mm_mul_ps(self.p1_, vs), _mm_mul_ps(self.p2_, vs))
        }
    }
}
impl Div<f64> for Line {
    type Output = Line;
    #[inline]
    fn div(self, s: f64) -> Self {
        self.div(s as f32)
    }
}
impl Div<i32> for Line {
    type Output = Line;
    #[inline]
    fn div(self, s: i32) -> Self {
        self.div(s as f32)
    }
}

/// Unary minus
impl Neg for Line {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        unsafe {
            let flip: __m128 = _mm_set1_ps(-0.);

            Self::from(_mm_xor_ps(self.p1_, flip), _mm_xor_ps(self.p2_, flip))
        }
    }
}

impl Line {
    #[inline]
    pub fn reverse(self) -> Line {
        unsafe {
            let flip: __m128 = _mm_set_ps(-0., -0., -0., 0.);
            Self::from(_mm_xor_ps(self.p1_, flip), _mm_xor_ps(self.p2_, flip))
        }
    }
}
