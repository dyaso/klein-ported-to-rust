#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sandwich::{sw012, sw02, sw312_four, sw32, swL2, swMM_four, swMM_three};
use crate::detail::sse::{dp_bc, rcp_nr1, rsqrt_nr1}; // hi_dp, hi_dp_bc, };

use crate::detail::exp_log::simd_exp;
use crate::detail::geometric_product::gpDL;

use crate::util::ApplyOp;
use crate::{Branch, Dual, IdealLine, Line, Plane, Point, Rotor, Translator};

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

impl Motor {
    pub fn default() -> Motor {
        unsafe {
            Motor {
                p1_: _mm_setzero_ps(),
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
    pub fn screw(ang_rad: f32, d: f32, l: Line) -> Motor {
        let mut log_m = Line::default();
        let mut out = Motor::default();
        gpDL(
            -ang_rad * 0.5,
            d * 0.5,
            l.p1_,
            l.p2_,
            &mut log_m.p1_,
            &mut log_m.p2_,
        );
        simd_exp(log_m.p1_, log_m.p2_, &mut out.p1_, &mut out.p2_);
        return out;
    }

    pub fn from_rotor(r: Rotor) -> Motor {
        unsafe {
            Motor {
                p1_: r.p1_,
                p2_: _mm_setzero_ps(),
            }
        }
    }

    pub fn from_translator(t: Translator) -> Motor {
        unsafe {
            Motor {
                p1_: _mm_set_ss(1.),
                p2_: t.p2_,
            }
        }
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
        return out;
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
        return out;
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
        return out;
    }

    pub fn scalar(self) -> f32 {
        let mut out: f32 = 0.;
        unsafe {
            _mm_store_ss(&mut out, self.p1_);
        }
        return out;
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

    pub fn e0123(self) -> f32 {
        let mut out: f32 = 0.;
        unsafe {
            _mm_store_ss(&mut out, self.p2_);
        }
        return out;
    }

    pub fn reverse(self) -> Motor {
        unsafe {
            let flip: __m128 = _mm_set_ps(-0., -0., -0., 0.);
            return Motor::from_rotor_and_translator(
                _mm_xor_ps(self.p1_, flip),
                _mm_xor_ps(self.p2_, flip),
            );
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

            return Self::from_rotor_and_translator(
                _mm_xor_ps(self.p1_, flip),
                _mm_xor_ps(self.p2_, flip),
            );
        }
    }
}

impl PartialEq for Motor {
    fn eq(&self, other: &Motor) -> bool {
        unsafe {
            let p1_eq: __m128 = _mm_cmpeq_ps(self.p1_, other.p1_);
            let p2_eq: __m128 = _mm_cmpeq_ps(self.p2_, other.p2_);
            let eq: __m128 = _mm_and_ps(p1_eq, p2_eq);
            return _mm_movemask_ps(eq) == 0xf;
        }
    }
}

impl ApplyOp<Plane> for Motor {
    fn apply_to(self, p: Plane) -> Plane {
        /// Conjugates a plane $p$ with this motor and returns the result
        /// $mp\widetilde{m}$.
        unsafe {
            return Plane::from(sw012(true, p.p0_, self.p1_, self.p2_));
        }
    }
}

impl ApplyOp<Point> for Motor {
    /// Conjugates a point $p$ with this motor and returns the result
    /// $mp\widetilde{m}$.
    fn apply_to(self, p: Point) -> Point {
        unsafe { return Point::from(sw312_four(true, p.p3_, self.p1_, self.p2_)) }
    }
}

/// Conjugates a line $\ell$ with this motor and returns the result
/// $m\ell \widetilde{m}$.
impl ApplyOp<Line> for Motor {
    fn apply_to(self, rhs: Line) -> Line {
        let (branch, ideal) = swMM_four(rhs.p1_, rhs.p2_, self.p1_, self.p2_);
        return Line::from(branch, ideal);
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use crate::{ApplyOp, EulerAngles, IdealLine, Line, Motor, Plane, Point, Rotor, Translator};

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
        let r = Rotor::rotor(pi * 0.5, 0., 0., 1.);
        let t = Translator::translator(1., 0., 0., 1.);
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

        let l: Line = m.log();
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
        let m = Motor::screw(pi * 0.5, 1., Line::new(0., 0., 0., 0., 0., 1.));
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
        let m = Motor::screw(
            pi * 0.5,
            3.,
            Line::new(3., 1., 2., 4., -2., 1.).normalized(),
        );

        let mut m2: Motor = m.sqrt();
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
