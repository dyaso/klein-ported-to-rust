#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sandwich::{sw012, sw012_six, sw312_four, sw312_six, sw_mm_four, sw_mm_seven};
use crate::detail::sse::{dp_bc, rcp_nr1, rsqrt_nr1}; // hi_dp, hi_dp_bc, };

use crate::detail::exp_log::simd_exp;
use crate::detail::geometric_product::gp_dl;

use crate::util::ApplyTo;
use crate::{ApplyToMany, Line, Plane, Point, Rotor, Translator};

/// \defgroup motor Motors
///
/// A `motor` represents a kinematic motion in our algebra. From
/// [Chasles'
/// theorem](https://en.wikipedia.org/wiki/Chasles%27_theorem_(kinematics)), we
/// know that any rigid body displacement can be produced by a translation along
/// a line, followed or preceded by a rotation about an axis parallel to that
/// line. The motor algebra is isomorphic to the dual quaternions but exists
/// here in the same algebra as all the other geometric entities and actions at
/// our disposal. Operations such as composing a motor with a rotor or
/// translator are possible for example. The primary benefit to using a motor
/// over its corresponding matrix operation is twofold. First, you get the
/// benefit of numerical stability when composing multiple actions via the
/// geometric product (`*`). Second, because the motors constitute a continuous
/// group, they are amenable to smooth interpolation and differentiation.
///
/// !!! example
///
/// ```c++
///     // Create a rotor representing a pi/2 rotation about the z-axis
///     // Normalization is done automatically
///     rotor r{kln::pi * 0.5f, 0.f, 0.f, 1.f};
///     
///     // Create a translator that represents a translation of 1 unit
///     // in the yz-direction. Normalization is done automatically.
///     translator t{1.f, 0.f, 1.f, 1.f};
///     
///     // Create a motor that combines the action of the rotation and
///     // translation above.
///     motor m = r * t;
///     
///     // Initialize a point at (1, 3, 2)
///     kln::point p1{1.f, 3.f, 2.f};
///     
///     // Translate p1 and rotate it to create a new point p2
///     kln::point p2 = m(p1);
/// ```
///
/// Motors can be multiplied to one another with the `*` operator to create
/// a new motor equivalent to the application of each factor.
///
/// !!! example
///
/// ```c++
///     // Suppose we have 3 motors m1, m2, and m3
///     
///     // The motor m created here represents the combined action of m1,
///     // m2, and m3.
///     kln::motor m = m3 * m2 * m1;
/// ```
///
/// The same `*` operator can be used to compose the motor's action with other
/// translators and rotors.
///
/// A demonstration of using the exponential and logarithmic map to blend
/// between two motors is provided in a test case
/// [here](https://github.com/jeremyong/Klein/blob/master/test/test_exp_log.cpp#L48).

/// \addtogroup motor
/// @{
/// \ingroup motor

#[derive(Copy, Clone, Debug)]
pub struct Motor {
    pub p1_: __m128,
    pub p2_: __m128,
}

use std::fmt;
impl fmt::Display for Motor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Motor\n\tScalar\te23\te31\te12\te01\te02\te03\te0123\n\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\n",
            self.scalar(),
            self.e23(),
            self.e32(),
            self.e12(),
            self.e01(),
            self.e02(),
            self.e03(),
            self.e0123()
        )
    }
}

impl From<Rotor> for Motor {
    fn from(rotor: Rotor) -> Self {
        unsafe {
            Motor {
                p1_: rotor.p1_,
                p2_: _mm_setzero_ps(),
            }
        }
    }
}

impl From<Translator> for Motor {
    fn from(translator: Translator) -> Self {
        unsafe {
            Motor {
                p1_: _mm_set_ss(1.),
                p2_: translator.p2_,
            }
        }
    }
}


impl Motor {
    pub fn default() -> Motor {
        unsafe {
            Motor {
                p1_: _mm_set_ss(1.),//_mm_setzero_ps(),
                p2_: _mm_setzero_ps(),
            }
        }
    }

    /// Direct initialization from components. A more common way of creating a
    /// motor is to take a product between a rotor and a translator.
    /// The arguments coorespond to the multivector
    /// $a + b\mathbf{e}_{23} + c\mathbf{e}_{31} + d\mathbf{e}_{12} +\
    /// e\mathbf{e}_{01} + f\mathbf{e}_{02} + g\mathbf{e}_{03} +\
    /// h\mathbf{e}_{0123}$.

    #[allow(clippy::too_many_arguments)] // i'm copying the c++ version!
    #[allow(clippy::many_single_char_names)]
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Motor {
        unsafe {
            Motor {
                p1_: _mm_set_ps(d, c, b, a),
                p2_: _mm_set_ps(g, f, e, h),
            }
        }
    }

    pub fn from_rotor_and_translator(rotor: __m128, translator: __m128) -> Motor {
        Motor {
            p1_: rotor,
            p2_: translator,
        }
    }

    /// Produce a screw motion rotating and translating by given amounts along a
    /// provided Euclidean axis.
    pub fn from_screw_line(ang_rad: f32, d: f32, l: Line) -> Motor {
        let mut log_m = Line::default();
        let mut out = Motor::default();
        println!("LOGin {}",log_m);
        gp_dl(
            -ang_rad * 0.5,
            d * 0.5,
            l.p1_,
            l.p2_,
            &mut log_m.p1_,
            &mut log_m.p2_,
        );
        println!("OUT {}",out);
        println!("LOG {}",log_m);
        simd_exp(log_m.p1_, log_m.p2_, &mut out.p1_, &mut out.p2_);
        println!("OUT {}",out);
        out
    }

    pub fn set_rotation(&mut self, r: Rotor) {
        unsafe {
            self.p1_ = r.p1_;
            self.p2_ = _mm_setzero_ps();
        }
    }

    pub fn set_translation(&mut self, t: Translator) {
        unsafe {
            self.p1_ = _mm_set_ss(1.);
            self.p2_ = t.p2_;
        }
    }

    /// Normalizes this motor $m$ such that $m\widetilde{m} = 1$.
    pub fn normalize(&mut self) {
        // m = b + c where b is p1 and c is p2
        //
        // m * ~m = |b|^2 + 2(b0 c0 - b1 c1 - b2 c2 - b3 c3)e0123
        //
        // The square root is given as:
        // |b| + (b0 c0 - b1 c1 - b2 c2 - b3 c3)/|b| e0123
        //
        // The inverse of this is given by:
        // 1/|b| + (-b0 c0 + b1 c1 + b2 c2 + b3 c3)/|b|^3 e0123 = s + t e0123
        //
        // Multiplying our original motor by this inverse will give us a
        // normalized motor.

        unsafe {
            let b2 = dp_bc(self.p1_, self.p1_);
            let s = rsqrt_nr1(b2);
            let bc = dp_bc(_mm_xor_ps(self.p1_, _mm_set_ss(-0.)), self.p2_);
            let t = _mm_mul_ps(_mm_mul_ps(bc, rcp_nr1(b2)), s);

            // (s + t e0123) * motor =
            //
            // s b0 +
            // s b1 e23 +
            // s b2 e31 +
            // s b3 e12 +
            // (s c0 + t b0) e0123 +
            // (s c1 - t b1) e01 +
            // (s c2 - t b2) e02 +
            // (s c3 - t b3) e03

            let tmp = _mm_mul_ps(self.p2_, s);
            self.p2_ = _mm_sub_ps(tmp, _mm_xor_ps(_mm_mul_ps(self.p1_, t), _mm_set_ss(-0.)));
            self.p1_ = _mm_mul_ps(self.p1_, s);
        }
    }

    /// Return a normalized copy of this motor.
    pub fn normalized_motor(self) -> Self {
        let mut out = Motor::clone(&self);
        out.normalize();
        out
    }

    /// Load motor data using two unaligned loads. This routine does *not*
    /// assume the data passed in this way is normalized.
    pub fn load(&mut self, source: &[f32; 2]) {
        // Aligned and unaligned loads incur the same amount of latency and have
        // identical throughput on most modern processors
        unsafe {
            self.p1_ = _mm_loadu_ps(&source[0]);
            self.p2_ = _mm_loadu_ps(&source[1]);
        }
    }

    pub fn invert(&mut self) {
        unsafe {
            // s, t computed as in the normalization
            let b2 = dp_bc(self.p1_, self.p1_);
            let s = rsqrt_nr1(b2);
            let bc = dp_bc(_mm_xor_ps(self.p1_, _mm_set_ss(-0.)), self.p2_);
            let b2_inv = rcp_nr1(b2);
            let t = _mm_mul_ps(_mm_mul_ps(bc, b2_inv), s);
            let neg = _mm_set_ps(-0., -0., -0., 0.);

            // p1 * (s + t e0123)^2 = (s * p1 - t p1_perp) * (s + t e0123)
            // = s^2 p1 - s t p1_perp - s t p1_perp
            // = s^2 p1 - 2 s t p1_perp
            // (the scalar component above needs to be negated)
            // p2 * (s + t e0123)^2 = s^2 p2 NOTE: s^2 = b2_inv
            let mut st = _mm_mul_ps(s, t);
            st = _mm_mul_ps(self.p1_, st);
            self.p2_ = _mm_sub_ps(
                _mm_mul_ps(self.p2_, b2_inv),
                _mm_xor_ps(_mm_add_ps(st, st), _mm_set_ss(-0.)),
            );
            self.p2_ = _mm_xor_ps(self.p2_, neg);

            self.p1_ = _mm_xor_ps(_mm_mul_ps(self.p1_, b2_inv), neg);
        }
    }

    pub fn inverse(self) -> Motor {
        let mut out = Motor::clone(&self);
        out.invert();
        out
    }

    /// Constrains the motor to traverse the shortest arc
    pub fn constrain(&mut self) {
        unsafe {
            let um = _mm_and_ps(self.p1_, _mm_set_ss(-0.));
            let mask: __m128 = _mm_shuffle_ps(um, um, 0);
            self.p1_ = _mm_xor_ps(mask, self.p1_);
            self.p2_ = _mm_xor_ps(mask, self.p2_);
        }
    }

    pub fn constrained(self) -> Motor {
        let mut out = Motor::clone(&self);
        out.constrain();
        out
    }

    pub fn scalar(self) -> f32 {
        let mut out: f32 = 0.;
        unsafe {
            _mm_store_ss(&mut out, self.p1_);
        }
        out
    }

    get_basis_blade_fn!(e12, e21, p1_, 3);
    get_basis_blade_fn!(e31, e13, p1_, 2);
    get_basis_blade_fn!(e23, e32, p1_, 1);
    get_basis_blade_fn!(e01, e10, p2_, 1);
    get_basis_blade_fn!(e02, e20, p2_, 2);
    get_basis_blade_fn!(e03, e30, p2_, 3);

    pub fn e0123(self) -> f32 {
        let mut out: f32 = 0.;
        unsafe {
            _mm_store_ss(&mut out, self.p2_);
        }
        out
    }

    pub fn reverse(self) -> Motor {
        unsafe {
            let flip: __m128 = _mm_set_ps(-0., -0., -0., 0.);
            Motor::from_rotor_and_translator(_mm_xor_ps(self.p1_, flip), _mm_xor_ps(self.p2_, flip))
        }
    }
}

use std::ops::Neg;

/// Unary minus
impl Neg for Motor {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        unsafe {
            let flip: __m128 = _mm_set1_ps(-0.);

            Self::from_rotor_and_translator(_mm_xor_ps(self.p1_, flip), _mm_xor_ps(self.p2_, flip))
        }
    }
}

use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Motor addition
impl AddAssign for Motor {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        unsafe {
            self.p1_ = _mm_add_ps(self.p1_, rhs.p1_);
            self.p2_ = _mm_add_ps(self.p2_, rhs.p2_);
        }
    }
}

/// Motor subtraction
impl SubAssign for Motor {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        unsafe {
            self.p1_ = _mm_sub_ps(self.p1_, rhs.p1_);
            self.p2_ = _mm_sub_ps(self.p2_, rhs.p2_);
        }
    }
}

/// Motor uniform scale
impl<T: Into<f32>> MulAssign<T> for Motor {
    //impl MulAssign<f32> for Motor {
    #[inline]
    fn mul_assign(&mut self, s: T) {
        unsafe {
            let vs: __m128 = _mm_set1_ps(s.into());
            self.p1_ = _mm_mul_ps(self.p1_, vs);
            self.p2_ = _mm_mul_ps(self.p2_, vs);
        }
    }
}

/// Motor uniform inverse scale
impl<T: Into<f32>> DivAssign<T> for Motor {
    #[inline]
    fn div_assign(&mut self, s: T) {
        unsafe {
            let vs: __m128 = rcp_nr1(_mm_set1_ps(s.into()));
            self.p1_ = _mm_mul_ps(self.p1_, vs);
            self.p2_ = _mm_mul_ps(self.p2_, vs);
        }
    }
}

use std::ops::{Add, Div, Mul, Sub};

impl Add for Motor {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Motor::from_rotor_and_translator(
                _mm_add_ps(self.p1_, rhs.p1_),
                _mm_add_ps(self.p2_, rhs.p2_),
            )
        }
    }
}

impl Sub for Motor {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Motor::from_rotor_and_translator(
                _mm_sub_ps(self.p1_, rhs.p1_),
                _mm_sub_ps(self.p2_, rhs.p2_),
            )
        }
    }
}

impl<T: Into<f32>> Mul<T> for Motor {
    type Output = Self;
    #[inline]
    fn mul(self, s: T) -> Self {
        unsafe {
            let vs: __m128 = _mm_set1_ps(s.into());
            Motor::from_rotor_and_translator(_mm_mul_ps(self.p1_, vs), _mm_mul_ps(self.p2_, vs))
        }
    }
}

/// Motor uniform inverse scale
impl<T: Into<f32>> Div<T> for Motor {
    type Output = Self;
    #[inline]
    fn div(self, s: T) -> Self {
        unsafe {
            let vs = rcp_nr1(_mm_set1_ps(s.into()));
            Motor::from_rotor_and_translator(_mm_mul_ps(self.p1_, vs), _mm_mul_ps(self.p2_, vs))
        }
    }
}

macro_rules! mul_scalar_by_motor {
    ($s:ty) => {
        impl Mul<Motor> for $s {
            type Output = Motor;
            #[inline]
            fn mul(self, l: Motor) -> Motor {
                l * (self as f32)
            }
        }
    };
}

mul_scalar_by_motor!(f32);
mul_scalar_by_motor!(f64);
mul_scalar_by_motor!(i32);

// /// Convert this motor to a 3x4 column-major matrix representing this
// /// motor's action as a linear transformation. The motor must be normalized
// /// for this conversion to produce well-defined results, but is more
// /// efficient than a 4x4 matrix conversion.
// [[nodiscard]] mat3x4 as_mat3x4() const noexcept
// {
//     mat3x4 out;
//     mat4x4_12<true, true>(p1_, &p2_, out.cols);
//     return out;
// }

// /// Convert this motor to a 4x4 column-major matrix representing this
// /// motor's action as a linear transformation.
// [[nodiscard]] mat4x4 as_mat4x4() const noexcept
// {
//     mat4x4 out;
//     mat4x4_12<true, false>(p1_, &p2_, out.cols);
//     return out;
// }

impl PartialEq for Motor {
    fn eq(&self, other: &Motor) -> bool {
        unsafe {
            let p1_eq: __m128 = _mm_cmpeq_ps(self.p1_, other.p1_);
            let p2_eq: __m128 = _mm_cmpeq_ps(self.p2_, other.p2_);
            let eq: __m128 = _mm_and_ps(p1_eq, p2_eq);
            _mm_movemask_ps(eq) == 0xf
        }
    }
}

impl ApplyTo<Plane> for Motor {
    /// Conjugates a plane $p$ with this motor and returns the result
    /// $mp\widetilde{m}$
    fn apply_to(self, p: Plane) -> Plane {
        Plane::from(sw012(true, p.p0_, self.p1_, self.p2_))
    }
}

// /// Conjugates a plane $p$ with this motor and returns the result
// /// $mp\widetilde{m}$.
// [[nodiscard]] plane KLN_VEC_CALL operator()(plane const& p) const noexcept
// {
// plane out;
// detail::sw012<false, true>(&p.p0_, p1_, &p2_, &out.p0_);
// return out;
// }

/// Conjugates an array of planes with this motor in the input array and
/// stores the result in the output array. Aliasing is only permitted when
/// `in == out` (in place motor application).
///
/// !!! tip
///
/// When applying a motor to a list of tightly packed planes, this
/// routine will be *significantly faster* than applying the motor to
/// each plane individually.

impl ApplyToMany<Plane> for Motor {
    fn apply_to_many(self, input: &[Plane], output: &mut [Plane], count: usize) {
        sw012_six(true, &input, self.p1_, self.p2_, output, count);
    }
}

impl ApplyTo<Point> for Motor {
    /// Conjugates a point $p$ with this motor and returns the result
    /// $mp\widetilde{m}$.
    fn apply_to(self, p: Point) -> Point {
        if p.w() == 0. {
            // it's a Direction, an 'ideal' point at infinity
            unsafe { Point::from(sw312_four(false, p.p3_, self.p1_, _mm_setzero_ps())) }
        } else {
            // it's just a regular point
            Point::from(sw312_four(true, p.p3_, self.p1_, self.p2_))
        }
    }
}

// Detail.sw312(false, &input.P3, P1, default, &p3, 1);

/// Conjugates a line $\ell$ with this motor and returns the result
/// $m\ell \widetilde{m}$.
impl ApplyTo<Line> for Motor {
    fn apply_to(self, rhs: Line) -> Line {
        let (branch, ideal) = sw_mm_four(rhs.p1_, rhs.p2_, self.p1_, self.p2_);
        Line::from(branch, ideal)
    }
}

/// Conjugates an array of points with this motor in the input array and
/// stores the result in the output array. Aliasing is only permitted when
/// `in == out` (in place motor application).
///
/// !!! tip
///
/// When applying a motor to a list of tightly packed points, this
/// routine will be *significantly faster* than applying the motor to
/// each point individually.
impl ApplyToMany<Point> for Motor {
    fn apply_to_many(self, input: &[Point], output: &mut [Point], count: usize) {
        sw312_six(true, &input, self.p1_, self.p2_, output, count);

        // unsafe {
        //     return Plane::from(sw012(true, p.p0_, self.p1_, self.p2_));
        // }
    }
}

/// Conjugates an array of lines with this motor in the input array and
/// stores the result in the output array. Aliasing is only permitted when
/// `in == out` (in place motor application).
///
/// !!! tip
///
/// When applying a motor to a list of tightly packed lines, this
/// routine will be *significantly faster* than applying the motor to
/// each line individually.
impl ApplyToMany<Line> for Motor {
    fn apply_to_many(self, input: &[Line], output: &mut [Line], count: usize) {
        sw_mm_seven(true, true, &input, self.p1_, self.p2_, output, count);
    }
}

#[cfg(test)]
mod tests {
    #![cfg(target_arch = "x86_64")]

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use crate::{ApplyTo, Line, Motor, Plane, Point, Rotor, Translator, log, sqrt};

    #[test]
    fn motor_plane() {
        let m = Motor::new(1., 4., 3., 2., 5., 6., 7., 8.);
        let p1 = Plane::new(3., 2., 1., -1.);
        let p2: Plane = m.apply_to(p1);
        assert_eq!(p2.x(), 78.);
        assert_eq!(p2.y(), 60.);
        assert_eq!(p2.z(), 54.);
        assert_eq!(p2.d(), 358.);
    }

    use crate::ApplyToMany;

    #[test]
    pub fn motor_plane_variadic() {
        let m = Motor::new(1., 4., 3., 2., 5., 6., 7., 8.);
        let ps: [Plane; 2] = [Plane::new(3., 2., 1., -1.), Plane::new(3., 2., 1., -1.)];
        let mut ps2 = <[Plane; 2]>::default();
        m.apply_to_many(&ps, &mut ps2, 2);

        for i in 0..2 {
            assert_eq!(ps2[i].x(), 78.);
            assert_eq!(ps2[i].y(), 60.);
            assert_eq!(ps2[i].z(), 54.);
            assert_eq!(ps2[i].d(), 358.);
        }
    }

    #[test]
    fn motor_point() {
        let m = Motor::new(1., 4., 3., 2., 5., 6., 7., 8.);
        let p1 = Point::new(-1., 1., 2.);
        let p2 = m.apply_to(p1);
        assert_eq!(p2.x(), -12.);
        assert_eq!(p2.y(), -86.);
        assert_eq!(p2.z(), -86.);
        assert_eq!(p2.w(), 30.);
    }

    #[test]
    fn motor_point_variadic() {
        let m = Motor::new(1., 4., 3., 2., 5., 6., 7., 8.);
        let ps: [Point; 2] = [Point::new(-1., 1., 2.), Point::new(-1., 1., 2.)];
        let mut ps2 = <[Point; 2]>::default();

        m.apply_to_many(&ps, &mut ps2, 2);

        for i in 0..2 {
            assert_eq!(ps2[i].x(), -12.);
            assert_eq!(ps2[i].y(), -86.);
            assert_eq!(ps2[i].z(), -86.);
            assert_eq!(ps2[i].w(), 30.);
        }
    }

    #[test]
    fn motor_constrain() {
        let mut m1 = Motor::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let mut m2 = m1.constrained();
        assert_eq!(m1, m2);

        m1 = -m1;
        m2 = m1.constrained();
        assert_eq!(m1, -m2);
    }

    #[test]
    fn construct_motor() {
        let pi = std::f32::consts::PI;
        let r = Rotor::new(pi * 0.5, 0., 0., 1.);
        let t = Translator::new(1., 0., 0., 1.);
        let mut m: Motor = r * t;
        let p1 = Point::new(1., 0., 0.);
        let mut p2: Point = m.apply_to(p1);
        assert_eq!(p2.x(), 0.);
        approx_eq(p2.y(), -1.);
        approx_eq(p2.z(), 1.);

        // Rotation and translation about the same axis commutes
        m = t * r;
        p2 = m.apply_to(p1);
        assert_eq!(p2.x(), 0.);
        approx_eq(p2.y(), -1.);
        approx_eq(p2.z(), 1.);

        let l: Line = log(m);
        assert_eq!(l.e23(), 0.);
        //CHECK_EQ(l.e12(), doctest::Approx(0.7854).epsilon(0.001));
        approx_eq(l.e12(), 0.785398); //.epsilon(0.001);
        assert_eq!(l.e31(), 0.);
        assert_eq!(l.e01(), 0.);
        assert_eq!(l.e02(), 0.);
        approx_eq(l.e03(), -0.5);
    }

    #[test]
    fn construct_motor_via_screw_axis() {
        let pi = std::f32::consts::PI;
        let m = Motor::from_screw_line(pi * 0.5, 1., Line::new(0., 0., 0., 0., 0., 1.));
        let p1 = Point::new(1., 0., 0.);
        let p2 = m.apply_to(p1);
        approx_eq(p2.x(), 0.);
        approx_eq(p2.y(), 1.);
        approx_eq(p2.z(), 1.);
    }

    #[test]
    fn motor_line() {
        let m = Motor::new(2., 4., 3., -1., -5., -2., 2., -3.);
        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(-1., 2., -3., -6., 5., 4.);
        let l2: Line = m.apply_to(l1);
        assert_eq!(l2.e01(), 6.);
        assert_eq!(l2.e02(), 522.);
        assert_eq!(l2.e03(), 96.);
        assert_eq!(l2.e12(), -214.);
        assert_eq!(l2.e31(), -148.);
        assert_eq!(l2.e23(), -40.);
    }

    #[test]
    fn motor_line_variadic() {
        let m = Motor::new(2., 4., 3., -1., -5., -2., 2., -3.);
        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let ls: [Line; 2] = [
            Line::new(-1., 2., -3., -6., 5., 4.),
            Line::new(-1., 2., -3., -6., 5., 4.),
        ];
        let mut ls2: [Line; 2] = [Line::default(), Line::default()];

        m.apply_to_many(&ls, &mut ls2, 2);

        for i in 0..2 {
            assert_eq!(ls2[i].e01(), 6.);
            assert_eq!(ls2[i].e02(), 522.);
            assert_eq!(ls2[i].e03(), 96.);
            assert_eq!(ls2[i].e12(), -214.);
            assert_eq!(ls2[i].e31(), -148.);
            assert_eq!(ls2[i].e23(), -40.);
        }
    }

    #[test]
    fn normalize_motor() {
        let mut m = Motor::new(1., 4., 3., 2., 5., 6., 7., 8.);
        m.normalize();
        let norm: Motor = m * m.reverse();
        approx_eq(norm.scalar(), 1.);
        approx_eq(norm.e0123(), 0.);
    }

    #[test]
    fn motor_sqrt() {
        let pi = std::f32::consts::PI;
        let m = Motor::from_screw_line(
            pi * 0.5,
            3.,
            Line::new(3., 1., 2., 4., -2., 1.).normalized(),
        );

        let mut m2: Motor = sqrt(m);
        m2 = m2 * m2;
        approx_eq(m.scalar(), m2.scalar());
        approx_eq(m.e01(), m2.e01());
        approx_eq(m.e02(), m2.e02());
        approx_eq(m.e03(), m2.e03());
        approx_eq(m.e23(), m2.e23());
        approx_eq(m.e31(), m2.e31());
        approx_eq(m.e12(), m2.e12());
        approx_eq(m.e0123(), m2.e0123());
    }
}
