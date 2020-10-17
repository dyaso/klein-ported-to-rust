#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{dp_bc, rcp_nr1, rsqrt_nr1}; // hi_dp, hi_dp_bc, };

//use crate::detail::exp_log::{simd_exp};

use crate::util::ApplyOp;
use crate::{Branch, Dual, IdealLine, Line, Motor, Plane, Point, Rotor, Translator};

/// \defgroup gp Geometric Product
///
/// The geometric product extends the exterior product with a notion of a
/// metric. When the subspace intersection of the operands of two basis
/// elements is non-zero, instead of the product extinguishing, the grade
/// collapses and a scalar weight is included in the final result according
/// to the metric. The geometric product can be used to build rotations, and
/// by extension, rotations and translations in projective space.
///
/// !!! example "Rotor composition"
///
/// ```cpp
///     kln::rotor r1{ang1, x1, y1, z1};
///     kln::rotor r2{ang2, x2, y2, z2};
///     
///     // Compose rotors with the geometric product
///     kln::rotor r3 = r1 * r2;; // r3 combines r2 and r1 in that order
/// ```
///
/// !!! example "Two reflections"
///
/// ```cpp
///     kln::plane p1{x1, y1, z1, d1};
///     kln::plane p2{x2, y2, z2, d2};
///     
///     // The geometric product of two planes combines their reflections
///     kln::motor m3 = p1 * p2; // m3 combines p2 and p1 in that order
///     // If p1 and p2 were parallel, m3 would be a translation. Otherwise,
///     // m3 would be a rotation.
/// ```
///
/// Another common usage of the geometric product is to create a transformation
/// that takes one entity to another. Suppose we have two entities $a$ and $b$
/// and suppose that both entities are normalized such that $a^2 = b^2 = 1$.
/// Then, the action created by $\sqrt{ab}$ is the action that maps $b$ to $a$.
///
/// !!! example "Motor between two lines"
///
/// ```cpp
///     kln::line l1{mx1, my1, mz1, dx1, dy1, dz1};
///     kln::line l2{mx2, my2, mz2, dx2, dy2, dz2};
///     // Ensure lines are normalized if they aren't already
///     l1.normalize();
///     l2.normalize();
///     kln::motor m = kln::sqrt(l1 * l2);
///     
///     kln::line l3 = m(l2);
///     // l3 will be projectively equivalent to l1.
/// ```
///
/// Also provided are division operators that multiply the first argument by the
/// inverse of the second argument.
/// \addtogroup gp
/// @{

/// Construct a motor $m$ such that $\sqrt{m}$ takes plane $b$ to plane $a$.
///
/// !!! example
///
/// ```cpp
///     kln::plane p1{x1, y1, z1, d1};
///     kln::plane p2{x2, y2, z2, d2};
///     kln::motor m = sqrt(p1 * p2);
///     plane p3 = m(p2);
///     // p3 will be approximately equal to p1
/// ```
//use crate::detail::sandwich::{sw02, swL2, sw32, sw012, swMM_three, sw312_four};
use crate::detail::geometric_product::{
    gp00, gp03_flip, gp03_noflip, gp11, gp12, gp33, gpLL, gpMM, gpRT,
};

use std::ops::{Div, Mul};

impl Mul<Plane> for Plane {
    type Output = Motor;
    #[inline]
    fn mul(self, p: Plane) -> Motor {
        let mut out = Motor::default();
        gp00(self.p0_, p.p0_, &mut out.p1_, &mut out.p2_);
        return out;
    }
}

impl Mul<Point> for Plane {
    type Output = Motor;
    #[inline]
    fn mul(self, p: Point) -> Motor {
        let mut out = Motor::default();
        gp03_noflip(self.p0_, p.p3_, &mut out.p1_, &mut out.p2_);
        return out;
    }
}

impl Mul<Plane> for Point {
    type Output = Motor;
    #[inline]
    fn mul(self, rhs: Plane) -> Motor {
        let mut out = Motor::default();
        gp03_flip(rhs.p0_, self.p3_, &mut out.p1_, &mut out.p2_);
        return out;
    }
}

impl Mul<Rotor> for Rotor {
    type Output = Rotor;
    #[inline]
    fn mul(self, rhs: Rotor) -> Rotor {
        let mut out = Rotor::default();
        gp11(self.p1_, rhs.p1_, &mut out.p1_);
        return out;
    }
}

/// Generates a motor $m$ that produces a screw motion about the common normal
/// to lines $a$ and $b$. The motor given by $\sqrt{m}$ takes $b$ to $a$
/// provided that $a$ and $b$ are both normalized.
impl Mul<Line> for Line {
    type Output = Motor;
    #[inline]
    fn mul(self, rhs: Line) -> Motor {
        let mut out = Motor::default();
        gpLL(
            self.p1_,
            self.p2_,
            rhs.p1_,
            rhs.p2_,
            &mut out.p1_,
            &mut out.p2_,
        );
        return out;
    }
}

/// Generates a translator $t$ that produces a displacement along the line
/// between points $a$ and $b$. The translator given by $\sqrt{t}$ takes $b$ to
/// $a$.
impl Mul<Point> for Point {
    type Output = Translator;
    #[inline]
    fn mul(self, rhs: Point) -> Translator {
        let mut out = Translator::default();
        out.p2_ = gp33(self.p3_, rhs.p3_);
        return out;
    }
}

// dual line

// line dual

/// Compose the action of a translator and rotor (`rhs` will be applied, then `self`)
impl Mul<Translator> for Rotor {
    type Output = Motor;
    #[inline]
    fn mul(self, rhs: Translator) -> Motor {
        let mut out = Motor::default();
        out.p1_ = self.p1_;
        out.p2_ = gpRT(false, self.p1_, rhs.p2_);
        return out;
    }
}

/// Compose the action of a rotor and translator (`rhs` will be applied, then `self`)
impl Mul<Rotor> for Translator {
    type Output = Motor;
    #[inline]
    fn mul(self, rhs: Rotor) -> Motor {
        let mut out = Motor::default();
        out.p1_ = rhs.p1_;
        out.p2_ = gpRT(true, rhs.p1_, self.p2_);
        //out.p2_ = gp12(false, self.p1_, rhs.p2_);
        return out;
    }
}

/// Compose the action of two translators (this operation is commutative for
/// these operands).
impl Mul<Translator> for Translator {
    type Output = Translator;
    #[inline]
    fn mul(self, rhs: Translator) -> Translator {
        return self + rhs;
    }
}

// impl Mul<Plane> for Point {
//     type Output = Plane;
//     #[inline]
//     fn mul(self, p: Plane) -> Self {
//         let mut out = Motor::default();
//         gp00(self.p0_, p.p0_, &mut out.p1_, &mut out.p2_);
//         return out
//     }
// }

/// Compose the action of a rotor and motor (`rhs` will be applied, then `self`)
impl Mul<Motor> for Rotor {
    type Output = Motor;
    #[inline]
    fn mul(self, rhs: Motor) -> Motor {
        let mut out = Motor::default();
        gp11(self.p1_, rhs.p1_, &mut out.p1_);
        out.p2_ = gp12(false, self.p1_, rhs.p2_);
        return out;
    }
}

/// Compose the action of a rotor and motor (`rhs` will be applied, then `self`)
impl Mul<Rotor> for Motor {
    type Output = Motor;
    #[inline]
    fn mul(self, rhs: Rotor) -> Motor {
        let mut out = Motor::default();
        gp11(self.p1_, rhs.p1_, &mut out.p1_);
        out.p2_ = gp12(true, rhs.p1_, self.p2_);
        return out;
    }
}

/// Compose the action of a translator and motor (`rhs` will be applied, then `self`)
impl Mul<Motor> for Translator {
    type Output = Motor;
    #[inline]
    fn mul(self, rhs: Motor) -> Motor {
        let mut out = Motor::default();
        out.p1_ = rhs.p1_;
        out.p2_ = gpRT(true, rhs.p1_, self.p2_);
        unsafe {
            out.p2_ = _mm_add_ps(out.p2_, rhs.p2_);
        }
        return out;
    }
}

/// Compose the action of a translator and motor (`rhs` will be applied, then `self`)
impl Mul<Translator> for Motor {
    type Output = Motor;
    #[inline]
    fn mul(self, rhs: Translator) -> Motor {
        let mut out = Motor::default();
        out.p1_ = self.p1_;
        out.p2_ = gpRT(false, self.p1_, rhs.p2_);
        unsafe {
            out.p2_ = _mm_add_ps(out.p2_, self.p2_);
        }
        return out;
    }
}

/// Compose the action of two motors (`rhs` will be applied, then `self`)
impl Mul<Motor> for Motor {
    type Output = Motor;
    #[inline]
    fn mul(self, rhs: Motor) -> Motor {
        let mut out = Motor::default();
        gpMM(
            self.p1_,
            self.p2_,
            rhs.p1_,
            rhs.p2_,
            &mut out.p1_,
            &mut out.p2_,
        );
        return out;
    }
}

// [[nodiscard]] inline motor KLN_VEC_CALL operator*(motor a, motor b) noexcept
// {
//     motor out;
//     detail::gpMM(a.p1_, b.p1_, &out.p1_);
//     return out;
// }

// Division operators

impl Div<Plane> for Plane {
    type Output = Motor;
    #[inline]
    fn div(self, rhs: Plane) -> Motor {
        return self * rhs.inverse();
    }
}
impl Div<Point> for Point {
    type Output = Translator;
    #[inline]
    fn div(self, rhs: Point) -> Translator {
        return self * rhs.inverse();
    }
}
impl Div<Branch> for Branch {
    type Output = Rotor;
    #[inline]
    fn div(self, rhs: Branch) -> Rotor {
        return self * rhs.inverse();
    }
}
// impl Div<Rotor> for Rotor {
//     type Output = Motor;
//     #[inline]
//     fn div(self, rhs: Rotor) -> Motor {
//         return self * rhs.inverse()
//     }
// }
impl Div<Translator> for Translator {
    type Output = Translator;
    #[inline]
    fn div(self, rhs: Translator) -> Translator {
        return self * rhs.inverse();
    }
}
impl Div<Line> for Line {
    type Output = Motor;
    #[inline]
    fn div(self, rhs: Line) -> Motor {
        return self * rhs.inverse();
    }
}
impl Div<Rotor> for Motor {
    type Output = Motor;
    #[inline]
    fn div(self, rhs: Rotor) -> Motor {
        return self * rhs.inverse();
    }
}
impl Div<Translator> for Motor {
    type Output = Motor;
    #[inline]
    fn div(self, rhs: Translator) -> Motor {
        return self * rhs.inverse();
    }
}
impl Div<Motor> for Motor {
    type Output = Motor;
    #[inline]
    fn div(self, rhs: Motor) -> Motor {
        return self * rhs.inverse();
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use crate::{
        ApplyOp, Branch, EulerAngles, IdealLine, Line, Motor, Plane, Point, Rotor, Translator,
    };

    #[test]
    fn multivector_gp_plane_plane() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let mut p1 = Plane::new(1., 2., 3., 4.);
        let p2 = Plane::new(2., 3., -1., -2.);
        let p12: Motor = p1 * p2;
        assert_eq!(p12.scalar(), 5.);
        assert_eq!(p12.e12(), -1.);
        assert_eq!(p12.e31(), 7.);
        assert_eq!(p12.e23(), -11.);
        assert_eq!(p12.e01(), 10.);
        assert_eq!(p12.e02(), 16.);
        assert_eq!(p12.e03(), 2.);
        assert_eq!(p12.e0123(), 0.);

        // plane p3 = sqrt(p1 / p2)(p2);
        // assert_eq!(p3.approx_eq(p1, 0.001f), true);

        p1.normalize();
        let m: Motor = p1 * p1;
        approx_eq(m.scalar(), 1.);
    }

    #[test]
    fn multivector_gp_plane_div_plane() {
        let p1 = Plane::new(1., 2., 3., 4.);
        let m: Motor = p1 / p1;
        approx_eq(m.scalar(), 1.);
        assert_eq!(m.e12(), 0.);
        assert_eq!(m.e31(), 0.);
        assert_eq!(m.e23(), 0.);
        assert_eq!(m.e01(), 0.);
        assert_eq!(m.e02(), 0.);
        assert_eq!(m.e03(), 0.);
        assert_eq!(m.e0123(), 0.);
    }

    #[test]
    fn multivector_gp_plane_point() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        // x*e_032 + y*e_013 + z*e_021 + e_123
        let p1 = Plane::new(1., 2., 3., 4.);
        let p2 = Point::new(-2., 1., 4.);

        let p1p2: Motor = p1 * p2;
        assert_eq!(p1p2.scalar(), 0.);
        assert_eq!(p1p2.e01(), -5.);
        assert_eq!(p1p2.e02(), 10.);
        assert_eq!(p1p2.e03(), -5.);
        assert_eq!(p1p2.e12(), 3.);
        assert_eq!(p1p2.e31(), 2.);
        assert_eq!(p1p2.e23(), 1.);
        assert_eq!(p1p2.e0123(), 16.);
    }

    #[test]
    fn line_normalization() {
        let mut l = Line::new(1., 2., 3., 3., 2., 1.);
        l.normalize();
        let m: Motor = l * l.reverse();
        approx_eq(m.scalar(), 1.);
        approx_eq(m.e23(), 0.);
        approx_eq(m.e31(), 0.);
        approx_eq(m.e12(), 0.);
        approx_eq(m.e01(), 0.);
        approx_eq(m.e02(), 0.);
        approx_eq(m.e03(), 0.);
        approx_eq(m.e0123(), 0.);
    }

    #[test]
    fn multivector_gp_line_line() {
        // a*e01 + b*e02 + c*e03 + d*e23 + e*e31 + f*e12
        let mut l1 = Line::new(1., 0., 0., 3., 2., 1.);
        let mut l2 = Line::new(0., 1., 0., 4., 1., -2.);

        let l1l2: Motor = l1 * l2;
        assert_eq!(l1l2.scalar(), -12.);
        assert_eq!(l1l2.e12(), 5.);
        assert_eq!(l1l2.e31(), -10.);
        assert_eq!(l1l2.e23(), 5.);
        assert_eq!(l1l2.e01(), 1.);
        assert_eq!(l1l2.e02(), -2.);
        assert_eq!(l1l2.e03(), -4.);
        assert_eq!(l1l2.e0123(), 6.);

        // l1.normalize();
        // l2.normalize();
        // line l3 = sqrt(l1 * l2)(l2);
        // CHECK_EQ(l3.approx_eq(-l1, 0.001f), true);
    }

    #[test]
    fn multivector_gp_branch_branch() {
        let b1 = Branch::new(2., 1., 3.);
        let b2 = Branch::new(1., -2., -3.);
        let r: Rotor = b2 * b1;
        assert_eq!(r.scalar(), 9.);
        assert_eq!(r.e23(), 3.);
        assert_eq!(r.e31(), 9.);
        assert_eq!(r.e12(), -5.);

        // b1.normalize();
        // b2.normalize();
        // branch b3 = ~sqrt(b2 * b1)(b1);
        // CHECK_EQ(b3.x(), doctest::Approx(b2.x()));
        // CHECK_EQ(b3.y(), doctest::Approx(b2.y()));
        // CHECK_EQ(b3.z(), doctest::Approx(b2.z()));
    }

    #[test]
    fn multivector_gp_branch_div_branch() {
        let b = Branch::new(2., 1., 3.);
        let r: Rotor = b / b;
        approx_eq(r.scalar(), 1.);
        assert_eq!(r.e23(), 0.);
        assert_eq!(r.e31(), 0.);
        assert_eq!(r.e12(), 0.);
    }

    #[test]
    fn multivector_gp_line_div_line() {
        let l = Line::new(1., -2., 2., -3., 3., -4.);
        let m: Motor = l / l;
        approx_eq(m.scalar(), 1.);
        assert_eq!(m.e12(), 0.);
        assert_eq!(m.e31(), 0.);
        assert_eq!(m.e23(), 0.);
        approx_eq(m.e01(), 0.);
        approx_eq(m.e02(), 0.);
        approx_eq(m.e03(), 0.);
        approx_eq(m.e0123(), 0.);
    }

    #[test]
    fn multivector_gp_point_plane() {
        // x*e_032 + y*e_013 + z*e_021 + e_123
        let p1 = Point::new(-2., 1., 4.);
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p2 = Plane::new(1., 2., 3., 4.);

        let p1p2: Motor = p1 * p2;
        assert_eq!(p1p2.scalar(), 0.);
        assert_eq!(p1p2.e01(), -5.);
        assert_eq!(p1p2.e02(), 10.);
        assert_eq!(p1p2.e03(), -5.);
        assert_eq!(p1p2.e12(), 3.);
        assert_eq!(p1p2.e31(), 2.);
        assert_eq!(p1p2.e23(), 1.);
        assert_eq!(p1p2.e0123(), -16.);
    }

    #[test]
    fn multivector_gp_point_point() {
        // x*e_032 + y*e_013 + z*e_021 + e_123
        let p1 = Point::new(1., 2., 3.);
        let p2 = Point::new(-2., 1., 4.);

        let p1p2: Translator = p1 * p2;
        approx_eq(p1p2.e01(), -3.);
        approx_eq(p1p2.e02(), -1.);
        approx_eq(p1p2.e03(), 1.);

        //     point p3 = sqrt(p1p2)(p2);
        //     CHECK_EQ(p3.x(), doctest::Approx(1.f));
        //     CHECK_EQ(p3.y(), doctest::Approx(2.f));
        //     CHECK_EQ(p3.z(), doctest::Approx(3.f));
    }

    #[test]
    fn multivector_gp_point_div_point() {
        let p1 = Point::new(1., 2., 3.);
        let t: Translator = p1 / p1;
        assert_eq!(t.e01(), 0.);
        assert_eq!(t.e02(), 0.);
        assert_eq!(t.e03(), 0.);
    }

    #[test]
    fn multivector_gp_translator_div_translator() {
        let t1 = Translator::translator(3., 1., -2., 3.);
        let t2: Translator = t1 / t1;
        assert_eq!(t2.e01(), 0.);
        assert_eq!(t2.e02(), 0.);
        assert_eq!(t2.e03(), 0.);
    }

    #[test]
    fn multivector_gp_rotor_translator() {
        let mut r = Rotor::default();
        unsafe {
            r.p1_ = _mm_set_ps(1., 0., 0., 1.);
        }
        let mut t = Translator::default();
        unsafe {
            t.p2_ = _mm_set_ps(1., 0., 0., 0.);
        }
        let m: Motor = r * t;
        assert_eq!(m.scalar(), 1.);
        assert_eq!(m.e01(), 0.);
        assert_eq!(m.e02(), 0.);
        assert_eq!(m.e03(), 1.);
        assert_eq!(m.e23(), 0.);
        assert_eq!(m.e31(), 0.);
        assert_eq!(m.e12(), 1.);
        assert_eq!(m.e0123(), 1.);
    }

    #[test]
    fn multivector_gp_translator_rotor() {
        let mut r = Rotor::default();
        unsafe {
            r.p1_ = _mm_set_ps(1., 0., 0., 1.);
        }
        let mut t = Translator::default();
        unsafe {
            t.p2_ = _mm_set_ps(1., 0., 0., 0.);
        }
        let m: Motor = t * r;
        assert_eq!(m.scalar(), 1.);
        assert_eq!(m.e01(), 0.);
        assert_eq!(m.e02(), 0.);
        assert_eq!(m.e03(), 1.);
        assert_eq!(m.e23(), 0.);
        assert_eq!(m.e31(), 0.);
        assert_eq!(m.e12(), 1.);
        assert_eq!(m.e0123(), 1.);
    }

    #[test]
    fn multivector_gp_motor_rotor() {
        let mut r1 = Rotor::default();
        unsafe {
            r1.p1_ = _mm_set_ps(1., 2., 3., 4.);
        }
        let mut t = Translator::default();
        unsafe {
            t.p2_ = _mm_set_ps(3., -2., 1., -3.);
        }
        let mut r2 = Rotor::default();
        unsafe {
            r2.p1_ = _mm_set_ps(-4., 2., -3., 1.);
        }
        let m1: Motor = (t * r1) * r2;
        let m2: Motor = t * (r1 * r2);
        assert_eq!(m1, m2);
    }

    #[test]
    fn multivector_gp_rotor_motor() {
        let mut r1 = Rotor::default();
        unsafe {
            r1.p1_ = _mm_set_ps(1., 2., 3., 4.);
        }
        let mut t = Translator::default();
        unsafe {
            t.p2_ = _mm_set_ps(3., -2., 1., -3.);
        }
        let mut r2 = Rotor::default();
        unsafe {
            r2.p1_ = _mm_set_ps(-4., 2., -3., 1.);
        }
        let m1: Motor = r2 * (r1 * t);
        let m2: Motor = (r2 * r1) * t;
        assert_eq!(m1, m2);
    }

    #[test]
    fn multivector_gp_motor_translator() {
        let mut r = Rotor::default();
        unsafe {
            r.p1_ = _mm_set_ps(1., 2., 3., 4.);
        }
        let mut t1 = Translator::default();
        unsafe {
            t1.p2_ = _mm_set_ps(3., -2., 1., -3.);
        }
        let mut t2 = Translator::default();
        unsafe {
            t2.p2_ = _mm_set_ps(-4., 2., -3., 1.);
        }
        let m1: Motor = (r * t1) * t2;
        let m2: Motor = r * (t1 * t2);
        assert_eq!(m1, m2);
    }

    #[test]
    fn multivector_gp_motor_motor() {
        let m1 = Motor::new(2., 3., 4., 5., 6., 7., 8., 9.);
        let m2 = Motor::new(6., 7., 8., 9., 10., 11., 12., 13.);
        let m3: Motor = m1 * m2;
        assert_eq!(m3.scalar(), -86.);
        assert_eq!(m3.e23(), 36.);
        assert_eq!(m3.e31(), 32.);
        assert_eq!(m3.e12(), 52.);
        assert_eq!(m3.e01(), -38.);
        assert_eq!(m3.e02(), -76.);
        assert_eq!(m3.e03(), -66.);
        assert_eq!(m3.e0123(), 384.);
    }

    #[test]
    fn multivector_gp_motor_div_motor() {
        let m1 = Motor::new(2., 3., 4., 5., 6., 7., 8., 9.);
        let m2: Motor = m1 / m1;
        approx_eq(m2.scalar(), 1.);
        assert_eq!(m2.e23(), 0.);
        assert_eq!(m2.e31(), 0.);
        assert_eq!(m2.e12(), 0.);
        assert_eq!(m2.e01(), 0.);
        approx_eq(m2.e02(), 0.);
        approx_eq(m2.e03(), 0.);
        approx_eq(m2.e0123(), 0.);
    }
}
