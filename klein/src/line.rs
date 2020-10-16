use crate::detail::sse::{hi_dp, dp_bc, hi_dp_bc, rcp_nr1, rsqrt_nr1};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

impl From<__m128> for IdealLine {
    fn from(xmm: __m128) -> Self {
        unsafe { IdealLine { p2_: xmm } }
    }
}

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
        return out;
    }

    pub fn ideal_norm(self) -> f32 {
        f32::sqrt(self.squared_ideal_norm())
    }
}

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Ideal line addition
impl AddAssign for IdealLine {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        unsafe { self.p2_ = _mm_add_ps(self.p2_, rhs.p2_) }
    }
}

/// Ideal line subtraction
impl SubAssign for IdealLine {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { self.p2_ = _mm_sub_ps(self.p2_, rhs.p2_) }
    }
}

/// Ideal line uniform scale
impl MulAssign<f32> for IdealLine {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        unsafe { self.p2_ = _mm_mul_ps(self.p2_, _mm_set1_ps(s)) }
    }
}
impl MulAssign<f64> for IdealLine {
    #[inline]
    fn mul_assign(&mut self, s: f64) {
        unsafe { self.p2_ = _mm_mul_ps(self.p2_, _mm_set1_ps(s as f32)) }
    }
}
impl MulAssign<i32> for IdealLine {
    #[inline]
    fn mul_assign(&mut self, s: i32) {
        unsafe { self.p2_ = _mm_mul_ps(self.p2_, _mm_set1_ps(s as f32)) }
    }
}

/// Ideal line uniform inverse scale
impl DivAssign<f32> for IdealLine {
    #[inline]
    fn div_assign(&mut self, s: f32) {
        unsafe { self.p2_ = _mm_mul_ps(self.p2_, rcp_nr1(_mm_set1_ps(s))) }
    }
}
impl DivAssign<f64> for IdealLine {
    #[inline]
    fn div_assign(&mut self, s: f64) {
        unsafe { self.p2_ = _mm_mul_ps(self.p2_, rcp_nr1(_mm_set1_ps(s as f32))) }
    }
}
impl DivAssign<i32> for IdealLine {
    #[inline]
    fn div_assign(&mut self, s: i32) {
        unsafe { self.p2_ = _mm_mul_ps(self.p2_, rcp_nr1(_mm_set1_ps(s as f32))) }
    }
}

/// Ideal line addition
impl Add for IdealLine {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { IdealLine::from(_mm_add_ps(self.p2_, rhs.p2_)) }
    }
}

/// Ideal line subtraction
impl Sub for IdealLine {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { IdealLine::from(_mm_sub_ps(self.p2_, rhs.p2_)) }
    }
}


//scalar_multiply!(IdealLine, unsafe { IdealLine::from(_mm_mul_ps(self.p2_, _mm_set1_ps(s))) });

/// Ideal line uniform scale
impl Mul<f32> for IdealLine {
    type Output = IdealLine;
    #[inline]
    fn mul(self, s: f32) -> Self {
        unsafe { IdealLine::from(_mm_mul_ps(self.p2_, _mm_set1_ps(s))) }
    }
}
impl Mul<IdealLine> for f32 {
    type Output = IdealLine;
    #[inline]
    fn mul(self, p: IdealLine) -> IdealLine {
        return p * self;
    }
}
impl Mul<f64> for IdealLine {
    type Output = IdealLine;
    #[inline]
    fn mul(self, s: f64) -> Self {
        unsafe { IdealLine::from(_mm_mul_ps(self.p2_, _mm_set1_ps(s as f32))) }
    }
}
impl Mul<IdealLine> for f64 {
    type Output = IdealLine;
    #[inline]
    fn mul(self, p: IdealLine) -> IdealLine {
        return p * self;
    }
}
impl Mul<i32> for IdealLine {
    type Output = IdealLine;
    #[inline]
    fn mul(self, s: i32) -> Self {
        unsafe { IdealLine::from(_mm_mul_ps(self.p2_, _mm_set1_ps(s as f32))) }
    }
}
impl Mul<IdealLine> for i32 {
    type Output = IdealLine;
    #[inline]
    fn mul(self, p: IdealLine) -> IdealLine {
        return p * self;
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
    pub fn e01(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p2_);
        }
        return out[1];
    }

    pub fn e10(self) -> f32 {
        -self.e01()
    }

    pub fn e02(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p2_);
        }
        return out[2];
    }

    pub fn e20(self) -> f32 {
        -self.e02()
    }

    pub fn e03(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p2_);
        }
        return out[3];
    }

    pub fn e30(self) -> f32 {
        -self.e03()
    }

    /// Reversion operator
    pub fn reverse(self) -> Self {
        unsafe {
            let flip: __m128 = _mm_set_ps(-0., -0., -0., 0.);
            Self::from(_mm_xor_ps(self.p2_, flip))
        }
    }
}

impl Neg for IdealLine {
    type Output = Self;
    /// Unary minus
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from(unsafe { _mm_xor_ps(self.p2_, _mm_set1_ps(-0.)) })
    }
}



































































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

impl From<__m128> for Branch {
    fn from(xmm: __m128) -> Branch {
        Branch { p1_: xmm } 
    }

}

impl Branch {
    pub fn default() -> Branch {
        unsafe{ Branch {p1_: _mm_setzero_ps()} }
    }

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
        return out;
    }

    /// Returns the square root of the quantity produced by `squared_norm`.
    pub fn norm(self) -> f32 {
        f32::sqrt(self.squared_norm())
    }

    pub fn normalize(&mut self) {
        unsafe {
            let mut inv_norm = _mm_setzero_ps();
            
            if self.scalar() == 0. {
                // it's a Branch
                inv_norm = rsqrt_nr1(hi_dp_bc(self.p1_, self.p1_));
            } else {
                // it's a Rotor
                inv_norm = rsqrt_nr1(dp_bc(self.p1_, self.p1_));
            }

            self.p1_ = _mm_mul_ps(self.p1_, inv_norm);
        }
    }

    pub fn normalized(self) -> Self {
        let mut out = Self::from(self.p1_);
        out.normalize();
        return out;
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
        return out;
    }
}

/// Branch addition
impl AddAssign for Branch {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        unsafe { self.p1_ = _mm_add_ps(self.p1_, rhs.p1_) }
    }
}

/// Branch subtraction
impl SubAssign for Branch {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { self.p1_ = _mm_sub_ps(self.p1_, rhs.p1_) }
    }
}

/// Branch uniform scale
impl MulAssign<f32> for Branch {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        unsafe { self.p1_ = _mm_mul_ps(self.p1_, _mm_set1_ps(s)) }
    }
}
impl MulAssign<f64> for Branch {
    #[inline]
    fn mul_assign(&mut self, s: f64) {
        unsafe { self.p1_ = _mm_mul_ps(self.p1_, _mm_set1_ps(s as f32)) }
    }
}
impl MulAssign<i32> for Branch {
    #[inline]
    fn mul_assign(&mut self, s: i32) {
        unsafe { self.p1_ = _mm_mul_ps(self.p1_, _mm_set1_ps(s as f32)) }
    }
}

/// Ideal line uniform inverse scale
impl DivAssign<f32> for Branch {
    #[inline]
    fn div_assign(&mut self, s: f32) {
        unsafe { self.p1_ = _mm_mul_ps(self.p1_, rcp_nr1(_mm_set1_ps(s))) }
    }
}
impl DivAssign<f64> for Branch {
    #[inline]
    fn div_assign(&mut self, s: f64) {
        unsafe { self.p1_ = _mm_mul_ps(self.p1_, rcp_nr1(_mm_set1_ps(s as f32))) }
    }
}
impl DivAssign<i32> for Branch {
    #[inline]
    fn div_assign(&mut self, s: i32) {
        unsafe { self.p1_ = _mm_mul_ps(self.p1_, rcp_nr1(_mm_set1_ps(s as f32))) }
    }
}

/// Branch addition
impl Add for Branch {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Branch::from(_mm_add_ps(self.p1_, rhs.p1_)) }
    }
}

/// Branch subtraction
impl Sub for Branch {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Branch::from(_mm_sub_ps(self.p1_, rhs.p1_)) }
    }
}

/// Branch uniform scale
impl Mul<f32> for Branch {
    type Output = Branch;
    #[inline]
    fn mul(self, s: f32) -> Self {
        unsafe { Branch::from(_mm_mul_ps(self.p1_, _mm_set1_ps(s))) }
    }
}
impl Mul<Branch> for f32 {
    type Output = Branch;
    #[inline]
    fn mul(self, p: Branch) -> Branch {
        return p * self;
    }
}
impl Mul<f64> for Branch {
    type Output = Branch;
    #[inline]
    fn mul(self, s: f64) -> Self {
        unsafe { Branch::from(_mm_mul_ps(self.p1_, _mm_set1_ps(s as f32))) }
    }
}
impl Mul<Branch> for f64 {
    type Output = Branch;
    #[inline]
    fn mul(self, p: Branch) -> Branch {
        return p * self;
    }
}
impl Mul<i32> for Branch {
    type Output = Branch;
    #[inline]
    fn mul(self, s: i32) -> Self {
        unsafe { Branch::from(_mm_mul_ps(self.p1_, _mm_set1_ps(s as f32))) }
    }
}
impl Mul<Branch> for i32 {
    type Output = Branch;
    #[inline]
    fn mul(self, p: Branch) -> Branch {
        return p * self;
    }
}

/// Ideal line uniform inverse scale
impl Div<f32> for Branch {
    type Output = Branch;
    #[inline]
    fn div(self, s: f32) -> Self {
        unsafe { Branch::from(_mm_mul_ps(self.p1_, rcp_nr1(_mm_set1_ps(s)))) }
    }
}
impl Div<f64> for Branch {
    type Output = Branch;
    #[inline]
    fn div(self, s: f64) -> Self {
        unsafe { Branch::from(_mm_mul_ps(self.p1_, rcp_nr1(_mm_set1_ps(s as f32)))) }
    }
}
impl Div<i32> for Branch {
    type Output = Branch;
    #[inline]
    fn div(self, s: i32) -> Self {
        unsafe { Branch::from(_mm_mul_ps(self.p1_, rcp_nr1(_mm_set1_ps(s as f32)))) }
    }
}

impl Branch {

    /// Store m128 contents into an array of 4 floats
    pub fn store(self) -> [f32;4] {
        let mut out = <[f32; 4]>::default();
        
        unsafe {_mm_store_ps(&mut out[0], self.p1_);}
        return out
    }

    pub fn e12(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p1_);
        }
        return out[3];
    }

    pub fn e21(self) -> f32 {
        -self.e12()
    }

    pub fn z(self) -> f32 {
        self.e12()
    }

    pub fn e31(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p1_);
        }
        return out[2];
    }

    pub fn e13(self) -> f32 {
        -self.e31()
    }

    pub fn y(self) -> f32 {
        self.e31()
    }

    pub fn e23(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p1_);
        }
        return out[1];
    }

    pub fn e32(self) -> f32 {
        -self.e23()
    }

    pub fn x(self) -> f32 {
        self.e23()
    }

    // rust port comment: Branches are also used to represent Rotors
    // Branches have scalar component = 0
    pub fn scalar(self) -> f32    {
        let mut out: f32 = 0.;
        unsafe {_mm_store_ss(&mut out, self.p1_);}
        return out
    }

    /// Reversion operator
    pub fn reverse(self) -> Self {
        unsafe {
            let flip: __m128 = _mm_set_ps(-0., -0., -0., 0.);
            Self::from(_mm_xor_ps(self.p1_, flip))
        }
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






































































impl Line {
    /// A line is specifed by 6 coordinates which correspond to the line's
    /// [Plücker
    /// coordinates](https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates).
    /// The coordinates specified in this way correspond to the following
    /// multivector:
    ///
    /// $$a\mathbf{e}_{01} + b\mathbf{e}_{02} + c\mathbf{e}_{03} +\
    /// d\mathbf{e}_{23} + e\mathbf{e}_{31} + f\mathbf{e}_{12}$$
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
                p1_: _mm_setzero_ps(),
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
                p1_: _mm_setzero_ps(),
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
        return out;
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
        return out;
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
        return out;
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
            return _mm_movemask_ps(cmp) == 0xf;
        }
    }
}

impl PartialEq for Line {
    fn eq(&self, other: &Line) -> bool {
        unsafe {
            let p1_eq: __m128 = _mm_cmpeq_ps(self.p1_, other.p1_);
            let p2_eq: __m128 = _mm_cmpeq_ps(self.p2_, other.p2_);
            let eq: __m128 = _mm_and_ps(p1_eq, p2_eq);
            return _mm_movemask_ps(eq) == 0xf;
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
impl MulAssign<f32> for Line {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        unsafe {
            let vs: __m128 = _mm_set1_ps(s);
            self.p1_ = _mm_mul_ps(self.p1_, vs);
            self.p2_ = _mm_mul_ps(self.p2_, vs);
        }
    }
}
impl MulAssign<f64> for Line {
    #[inline]
    fn mul_assign(&mut self, s: f64) {
        self.mul_assign(s as f32)
    }
}
impl MulAssign<i32> for Line {
    #[inline]
    fn mul_assign(&mut self, s: i32) {
        self.mul_assign(s as f32)
    }
}

/// Line uniform inverse scale
impl DivAssign<f32> for Line {
    #[inline]
    fn div_assign(&mut self, s: f32) {
        unsafe {
            let vs: __m128 = rcp_nr1(_mm_set1_ps(s));
            self.p1_ = _mm_mul_ps(self.p1_, vs);
            self.p2_ = _mm_mul_ps(self.p2_, vs);
        }
    }
}
impl DivAssign<f64> for Line {
    #[inline]
    fn div_assign(&mut self, s: f64) {
        self.div_assign(s as f32)
    }
}
impl DivAssign<i32> for Line {
    #[inline]
    fn div_assign(&mut self, s: i32) {
        self.div_assign(s as f32)
    }
}

impl Line {
    pub fn e12(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p1_);
        }
        return out[3];
    }

    pub fn e21(self) -> f32 {
        -self.e12()
    }

    pub fn e31(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p1_);
        }
        return out[2];
    }

    pub fn e13(self) -> f32 {
        -self.e31()
    }

    pub fn e23(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p1_);
        }
        return out[1];
    }

    pub fn e32(self) -> f32 {
        -self.e23()
    }

    pub fn e01(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p2_);
        }
        return out[1];
    }

    pub fn e10(self) -> f32 {
        -self.e01()
    }

    pub fn e02(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p2_);
        }
        return out[2];
    }

    pub fn e20(self) -> f32 {
        -self.e02()
    }

    pub fn e03(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p2_);
        }
        return out[3];
    }

    pub fn e30(self) -> f32 {
        -self.e03()
    }
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
impl Mul<f32> for Line {
    type Output = Line;
    #[inline]
    fn mul(self, s: f32) -> Self {
        unsafe {
            let vs: __m128 = _mm_set1_ps(s);
            Line::from(_mm_mul_ps(self.p1_, vs), _mm_mul_ps(self.p2_, vs))
        }
    }
}
impl Mul<f64> for Line {
    type Output = Line;
    #[inline]
    fn mul(self, s: f64) -> Self {
        self.mul(s as f32)
    }
}
impl Mul<i32> for Line {
    type Output = Line;
    #[inline]
    fn mul(self, s: i32) -> Self {
        self.mul(s as f32)
    }
}

impl Mul<Line> for f32 {
    type Output = Line;
    #[inline]
    fn mul(self, l: Line) -> Line {
        return l * self;
    }
}
impl Mul<Line> for f64 {
    type Output = Line;
    #[inline]
    fn mul(self, l: Line) -> Line {
        return l * self as f32;
    }
}
impl Mul<Line> for i32 {
    type Output = Line;
    #[inline]
    fn mul(self, l: Line) -> Line {
        return l * self as f32;
    }
}

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

            return Self::from(_mm_xor_ps(self.p1_, flip), _mm_xor_ps(self.p2_, flip));
        }
    }
}

impl Line {
    #[inline]
    pub fn reverse(self) -> Line {
        unsafe {
            let flip: __m128 = _mm_set_ps(-0., -0., -0., 0.);
            return Self::from(_mm_xor_ps(self.p1_, flip), _mm_xor_ps(self.p2_, flip));
        }
    }
}
