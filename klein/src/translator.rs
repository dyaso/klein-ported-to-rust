#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{dp_bc, hi_dp, hi_dp_bc, rcp_nr1, rsqrt_nr1};
use crate::detail::sandwich::{sw02, sw32, sw_l2};

use crate::util::ApplyTo;
use crate::{IdealLine, Line, Plane, Point};

/// \defgroup translator Translators
///
/// A translator represents a rigid-body displacement along a normalized axis.
/// To apply the translator to a supported entity, the call operator is
/// available.
///
/// !!! example
///
/// ```c++
///     // Initialize a point at (1, 3, 2)
///     kln::point p{1.f, 3.f, 2.f};
///      
///     // Create a normalized translator representing a 4-unit
///     // displacement along the xz-axis.
///     kln::translator r{4.f, 1.f, 0.f, 1.f};
///     
///     // Displace our point using the created translator
///     kln::point translated = r(p);
/// ```
/// We can translate lines and planes as well using the translator's call
/// operator.
///
/// Translators can be multiplied to one another with the `*` operator to create
/// a new translator equivalent to the application of each factor.
///
/// !!! example
///
/// ```c++
///     // Suppose we have 3 translators t1, t2, and t3
///     
///     // The translator t created here represents the combined action of
///     // t1, t2, and t3.
///     kln::translator t = t3 * t2 * t1;
/// ```
///
/// The same `*` operator can be used to compose the translator's action with
/// other rotors and motors.

//pub type Translator = IdealLine;

#[derive(Copy, Clone)]
pub struct Translator {
    pub p2_: __m128
}

impl Translator {
    pub fn translator(delta: f32, x: f32, y: f32, z: f32) -> Translator {
        let norm = f32::sqrt(x * x + y * y + z * z);
        let inv_norm: f32 = 1. / norm;

        let half_d: f32 = -0.5 * delta;
        unsafe {
            let mut p2_ = _mm_mul_ps(_mm_set1_ps(half_d), _mm_set_ps(z, y, x, 0.));
            p2_ = _mm_mul_ps(p2_, _mm_set_ps(inv_norm, inv_norm, inv_norm, 0.));
            Translator::from(p2_)
        }
    }

    pub fn load_normalized(&mut self, data: &f32) {
        unsafe {
            self.p2_ = _mm_loadu_ps(data);
        }
    }

    pub fn default() -> Translator {
        unsafe { Translator::from(_mm_setzero_ps()) }
    }

    pub fn invert(&mut self) {
        unsafe {
            self.p2_ = _mm_xor_ps(_mm_set_ps(-0., -0., -0., 0.), self.p2_);
        }
    }

    pub fn normalize(self) -> Self {
        unimplemented!()
    }

    pub fn inverse(self) -> Translator {
        let mut out = Translator::from(self.p2_);
        out.invert();
        out
    }
}

/// Conjugates a plane $p$ with this translator and returns the result
/// $tp\widetilde{t}$.
impl ApplyTo<Plane> for Translator {
    fn apply_to(self, p: Plane) -> Plane {
        unsafe {
            let tmp: __m128;
            if is_x86_feature_detected!("sse4.1") {
                tmp = _mm_blend_ps(self.p2_, _mm_set_ss(1.), 1);
            } else {
                tmp = _mm_add_ps(self.p2_, _mm_set_ss(1.));
            }
            Plane::from(sw02(p.p0_, tmp))
        }
    }
}

common_operations!(Translator, p2_);

impl Translator {
    get_basis_blade_fn!(e01, e10, p2_, 1);
    get_basis_blade_fn!(e02, e20, p2_, 2);
    get_basis_blade_fn!(e03, e30, p2_, 3);

}

/// Conjugates a line $\ell$ with this translator and returns the result
/// $t\ell\widetilde{t}$.
impl ApplyTo<Line> for Translator {
    fn apply_to(self, l: Line) -> Line {
        Line::from(l.p1_, sw_l2(l.p1_, l.p2_, self.p2_))
    }
}

/// Conjugates a point $p$ with this translator and returns the result
/// $tp\widetilde{t}$.
impl ApplyTo<Point> for Translator {
    fn apply_to(self, p: Point) -> Point {
        Point::from(sw32(p.p3_, self.p2_))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[allow(dead_code)]
    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use crate::{ApplyTo, Line, Point, Translator};

    #[test]
    fn translator_point() {
        let t = Translator::translator(1., 0., 0., 1.);
        let p1 = Point::new(1., 0., 0.);
        let p2: Point = t.apply_to(p1);
        assert_eq!(p2.x(), 1.);
        assert_eq!(p2.y(), 0.);
        assert_eq!(p2.z(), 1.);
    }

    #[test]
    fn translator_line() {
        let data: [f32; 4] = [0., -5., -2., 2.];
        let mut t = Translator::default();
        t.load_normalized(&data[0]);
        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(-1., 2., -3., -6., 5., 4.);
        let l2: Line = t.apply_to(l1);
        assert_eq!(l2.e01(), 35.);
        assert_eq!(l2.e02(), -14.);
        assert_eq!(l2.e03(), 71.);
        assert_eq!(l2.e12(), 4.);
        assert_eq!(l2.e31(), 5.);
        assert_eq!(l2.e23(), -6.);
    }

    use crate::detail::sandwich::sw02;
    #[test]
    fn simd_sandwich() {
        let mut ab = <[f32; 4]>::default();

        unsafe {
            let a = _mm_set_ps(4., 3., 2., 1.);
            let b = _mm_set_ps(-1., -2., -3., -4.);
            _mm_store_ps(&mut ab[0], sw02(a, b));
        }

        assert_eq!(ab[0], 9.);
        assert_eq!(ab[1], 2.);
        assert_eq!(ab[2], 3.);
        assert_eq!(ab[3], 4.);
    }
}
