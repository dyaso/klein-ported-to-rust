#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::exterior_product::{ext00, ext02, ext03_false, ext03_true, ext_pb};
use crate::detail::sse::hi_dp_ss; //rcp_nr1, hi_dp, hi_dp_bc, rsqrt_nr1};


use crate::{Branch, Dual, IdealLine, Line, Plane, Point};
//use crate::plane::Plane;

/// Exterior Product
///
/// (meet.hpp)
///
/// The exterior product between two basis elements extinguishes if the two
/// operands share any common index. Otherwise, the element produced is
/// equivalent to the union of the subspaces. A sign flip is introduced if
/// the concatenation of the element indices is an odd permutation of the
/// cyclic basis representation. The exterior product extends to general
/// multivectors by linearity.
///
/// !!! example "Meeting two planes"
///
/// ```cpp
///     kln::plane p1{x1, y1, z1, d1};
///     kln::plane p2{x2, y2, z2, d2};
///     // l lies at the intersection of p1 and p2.
///     kln::line l = p1 ^ p2;
/// ```
///
/// !!! example "Meeting a line and a plane"
///
/// ```cpp
///     kln::plane p1{x, y, z, d};
///     kln::line l2{mx, my, mz, dx, dy, dz};
///  
///     // p2 lies at the intersection of p1 and l2.
///     kln::point p2 = p1 ^ l2;
/// ```
use std::ops::BitXor;   // |

impl BitXor<Plane> for Plane {
    type Output = Line;
    #[inline]
    fn bitxor(self, rhs: Plane) -> Self::Output {
        let mut out = Line::default();
        ext00(self.p0_, rhs.p0_, &mut out.p1_, &mut out.p2_);
        return out;
    }
}

impl BitXor<Branch> for Plane {
    type Output = Point;
    #[inline]
    fn bitxor(self, rhs: Branch) -> Self::Output {
        let mut out = Point::default();
        out.p3_ = ext_pb(self.p0_, rhs.p1_);
        return out;
    }
}
impl BitXor<Plane> for Branch {
    type Output = Point;
    #[inline]
    fn bitxor(self, rhs: Plane) -> Self::Output {
        rhs.bitxor(self)
    }
}

impl BitXor<Line> for Plane {
    type Output = Point;
    #[inline]
    fn bitxor(self, rhs: Line) -> Self::Output {
        let mut out = Point::default();
        out.p3_ = ext_pb(self.p0_, rhs.p1_);
        let tmp = ext02(self.p0_, rhs.p2_);
        unsafe {
            out.p3_ = _mm_add_ps(tmp, out.p3_);
        }
        return out
    }
}
impl BitXor<Plane> for Line {
    type Output = Point;
    #[inline]
    fn bitxor(self, rhs: Plane) -> Self::Output {
        rhs.bitxor(self)
    }
}

impl BitXor<Point> for Plane {
    type Output = Dual;
    #[inline]
    fn bitxor(self, rhs: Point) -> Self::Output {
        let mut out = Dual::default();
        let tmp = ext03_false(self.p0_, rhs.p3_);
        unsafe {
            _mm_store_ss(&mut out.q, tmp);
        }
        return out;
    }
}

impl BitXor<Plane> for Point {
    type Output = Dual;
    #[inline]
    fn bitxor(self, rhs: Plane) -> Self::Output {
        let mut out = Dual::default();
        let tmp = ext03_true(rhs.p0_, self.p3_);
        unsafe {
            _mm_store_ss(&mut out.q, tmp);
        }
        return out;
    }
}

impl BitXor<IdealLine> for Plane {
    type Output = Point;
    #[inline]
    fn bitxor(self, rhs: IdealLine) -> Self::Output {
        let mut out = Point::default();
        out.p3_ = ext02(self.p0_, rhs.p2_);
        return out;
    }
}
impl BitXor<Plane> for IdealLine {
    type Output = Point;
    #[inline]
    fn bitxor(self, rhs: Plane) -> Self::Output {
        rhs.bitxor(self)
    }
}

impl BitXor<IdealLine> for Branch {
    type Output = Dual;
    #[inline]
    fn bitxor(self, rhs: IdealLine) -> Self::Output {
        let mut out = Dual::default();
        let tmp = hi_dp_ss(self.p1_, rhs.p2_);
        unsafe {
            _mm_store_ss(&mut out.q, tmp);
        }
        return out;
    }
}
impl BitXor<Branch> for IdealLine {
    type Output = Dual;
    #[inline]
    fn bitxor(self, rhs: Branch) -> Self::Output {
        rhs.bitxor(self)
    }
}

impl BitXor<Line> for Line {
    type Output = Dual;
    #[inline]
    fn bitxor(self, rhs: Line) -> Self::Output {
        unsafe {
            let mut dp: __m128 = hi_dp_ss(self.p1_, rhs.p2_);
            let mut out1 = f32::default();
            _mm_store_ss(&mut out1, dp);
            dp = hi_dp_ss(rhs.p1_, self.p2_);
            let mut out2 = f32::default();
            _mm_store_ss(&mut out2, dp);
            return Dual {
                p: 0.,
                q: out1 + out2,
            };
        }
    }
}

impl BitXor<IdealLine> for Line {
    type Output = Dual;
    #[inline]
    fn bitxor(self, rhs: IdealLine) -> Self::Output {
        return Branch::from(self.p1_) ^ rhs;
    }
}
impl BitXor<Line> for IdealLine {
    type Output = Dual;
    #[inline]
    fn bitxor(self, rhs: Line) -> Self::Output {
        return rhs ^ self;
    }
}

impl BitXor<Branch> for Line {
    type Output = Dual;
    #[inline]
    fn bitxor(self, rhs: Branch) -> Self::Output {
        return IdealLine::from(self.p2_) ^ rhs;
    }
}

impl BitXor<Line> for Branch {
    type Output = Dual;
    #[inline]
    fn bitxor(self, rhs: Line) -> Self::Output {
        return rhs ^ self;
    }
}

#[cfg(test)]
mod tests {
    #![cfg(target_arch = "x86_64")]

    #[allow(dead_code)]
    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use crate::{Dual, IdealLine, Line, Plane, Point};

    #[test]
    fn multivector_ep_plane_plane() {
        let p1 = Plane::new(1., 2., 3., 4.);
        let p2 = Plane::new(2., 3., -1., -2.);
        let p12: Line = p1 ^ p2;

        assert_eq!(p12.e01(), 10.);
        assert_eq!(p12.e02(), 16.);
        assert_eq!(p12.e03(), 2.);
        assert_eq!(p12.e12(), -1.);
        assert_eq!(p12.e31(), 7.);
        assert_eq!(p12.e23(), -11.);
    }

    #[test]
    fn multivector_ep_plane_line() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);

        // a*e01 + b*e02 + c*e03 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(0., 0., 1., 4., 1., -2.);

        let p1l1: Point = p1 ^ l1;

        assert_eq!(p1l1.e021(), 8.);
        assert_eq!(p1l1.e013(), -5.);
        assert_eq!(p1l1.e032(), -14.);
        assert_eq!(p1l1.e123(), 0.);
    }

    #[test]
    fn multivector_ep_plane_ideal_line() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);

        // a*e01 + b*e02 + c*e03
        let l1 = IdealLine::new(-2., 1., 4.);

        let p1l1: Point = p1 ^ l1;
        assert_eq!(p1l1.e021(), 5.);
        assert_eq!(p1l1.e013(), -10.);
        assert_eq!(p1l1.e032(), 5.);
        assert_eq!(p1l1.e123(), 0.);
    }

    #[test]
    fn multivector_ep_plane_point() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);
        // x*e_032 + y*e_013 + z*e_021 + e_123
        let p2 = Point::new(-2., 1., 4.);

        let p1p2: Dual = p1 ^ p2;
        assert_eq!(p1p2.scalar(), 0.);
        assert_eq!(p1p2.e0123(), 16.);
    }

    #[test]
    fn multivector_ep_line_plane() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);

        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(0., 0., 1., 4., 1., -2.);

        let p1l1 = l1 ^ p1;
        assert_eq!(p1l1.e021(), 8.);
        assert_eq!(p1l1.e013(), -5.);
        assert_eq!(p1l1.e032(), -14.);
        assert_eq!(p1l1.e123(), 0.);
    }

    #[test]
    fn multivector_ep_line_line() {
        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(1., 0., 0., 3., 2., 1.);
        let l2 = Line::new(0., 1., 0., 4., 1., -2.);

        let l1l2: Dual = l1 ^ l2;
        assert_eq!(l1l2.e0123(), 6.);
    }

    #[test]
    fn multivector_ep_ideal_line_plane() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);

        // a*e01 + b*e02 + c*e03
        let l1 = IdealLine::new(-2., 1., 4.);

        let p1l1 = l1 ^ p1;
        assert_eq!(p1l1.e021(), 5.);
        assert_eq!(p1l1.e013(), -10.);
        assert_eq!(p1l1.e032(), 5.);
        assert_eq!(p1l1.e123(), 0.);
    }

    #[test]
    fn multivector_ep_point_plane() {
        // x*e_032 + y*e_013 + z*e_021 + e_123
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Point::new(-2., 1., 4.);
        let p2 = Plane::new(1., 2., 3., 4.);

        let p1p2: Dual = p1 ^ p2;
        assert_eq!(p1p2.e0123(), -16.);
    }
}
