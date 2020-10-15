#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::hi_dp_ss; //rcp_nr1, hi_dp, hi_dp_bc, rsqrt_nr1};
use crate::detail::inner_product::{dot00, dotPL_noflip, dotPL_flip, dotPIL_noflip, dotPIL_flip, dot03, dot11, dotPTL, dot33};

use crate::{Plane, Point, Line, IdealLine, Branch, Dual};

use std::ops::BitOr;

/// \defgroup dot Symmetric Inner Product
///
/// The symmetric inner product takes two arguments and contracts the lower
/// graded element to the greater graded element. If lower graded element
/// spans an index that is not contained in the higher graded element, the
/// result is annihilated. Otherwise, the result is the part of the higher
/// graded element "most unlike" the lower graded element. Thus, the
/// symmetric inner product can be thought of as a bidirectional contraction
/// operator.
///
/// There is some merit in providing both a left and right contraction
/// operator for explicitness. However, when using Klein, it's generally
/// clear what the interpretation of the symmetric inner product is with
/// respect to the projection on various entities.
///
/// !!! example "Angle between planes"
///
/// ```cpp
///     kln::plane a{x1, y1, z1, d1};
///     kln::plane b{x2, y2, z2, d2};
///
///     // Compute the cos of the angle between two planes
///     float cos_ang = a | b;
/// ```
///
/// !!! example "Line to plane through point"
///
/// ```cpp
///     kln::point a{x1, y1, z1};
///     kln::plane b{x2, y2, z2, d2};
///	
///     // The line l contains a and the shortest path from a to plane b.
///     line l = a | b;
/// ```

impl BitOr<Plane> for Plane {
    type Output = f32;
    #[inline]
    fn bitor(self, rhs: Plane) -> Self::Output {
	    let mut out:f32 = 0.;
	    let s = dot00(self.p0_, rhs.p0_);
	    unsafe {_mm_store_ss(&mut out, s);}
	    return out
    }
}

impl BitOr<Line> for Plane {
    type Output = Plane;
    #[inline]
    fn bitor(self, rhs: Line) -> Self::Output {
	    return Plane::from(dotPL_noflip(self.p0_, rhs.p1_, rhs.p2_))
	}
}

impl BitOr<Plane> for Line {
    type Output = Plane;
    #[inline]
    fn bitor(self, rhs: Plane) -> Self::Output {
    	return Plane::from(dotPL_flip(rhs.p0_, self.p1_, self.p2_))
    }
}

impl BitOr<IdealLine> for Plane {
    type Output = Plane;
    #[inline]
    fn bitor(self, rhs: IdealLine) -> Self::Output {
	    return Plane::from(dotPIL_noflip(self.p0_, rhs.p2_))
	}
}

impl BitOr<Plane> for IdealLine {
    type Output = Plane;
    #[inline]
    fn bitor(self, rhs: Plane) -> Self::Output {
    	return Plane::from(dotPIL_flip(rhs.p0_, self.p2_))
    }
}



impl BitOr<Point> for Plane {
    type Output = Line;
    #[inline]
    fn bitor(self, rhs: Point) -> Self::Output {
    	let mut out = Line::default();
    	dot03(self.p0_, rhs.p3_, &mut out.p1_, &mut out.p2_);
	    return out
	}
}

impl BitOr<Plane> for Point {
    type Output = Line;
    #[inline]
    fn bitor(self, rhs: Plane) -> Self::Output {
    	return rhs | self
    }
}

impl BitOr<Line> for Line {
    type Output = f32;
    #[inline]
    fn bitor(self, rhs: Line) -> Self::Output {
    	let mut out:f32 = 0.;
    	unsafe {_mm_store_ss(&mut out, dot11(self.p1_, rhs.p1_)); }
    	return out
    }
}



impl BitOr<Line> for Point {
    type Output = Plane;
    #[inline]
    fn bitor(self, rhs: Line) -> Self::Output {
    	return Plane::from(dotPTL(self.p3_, rhs.p1_))
    }
}

impl BitOr<Point> for Line {
    type Output = Plane;
    #[inline]
    fn bitor(self, rhs: Point) -> Self::Output {
    	return rhs | self
    }
}

impl BitOr<Point> for Point {
    type Output = f32;
    #[inline]
    fn bitor(self, rhs: Point) -> Self::Output {
    	let mut out: f32 = 0.;
    	unsafe {_mm_store_ss(&mut out, dot33(self.p3_, rhs.p3_))}
    	return out
    }
}


































#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use crate::{Line, IdealLine, Plane, Point};

    #[test]
    fn multivector_ip_plane_plane() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);
        let p2 = Plane::new(2., 3., -1., -2.);
        let p12:f32 = p1 | p2;
        assert_eq!(p12, 5.);
    }


    #[test]
    fn multivector_ip_plane_line() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);

        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(0., 0., 1., 4., 1., -2.);

        let p1l1:Plane = p1 | l1;
        assert_eq!(p1l1.e0(), -3.);
        assert_eq!(p1l1.e1(), 7.);
        assert_eq!(p1l1.e2(), -14.);
        assert_eq!(p1l1.e3(), 7.);
	}

    #[test]
    fn multivector_ip_line_plane() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);

        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(0., 0., 1., 4., 1., -2.);

        let p1l1:Plane = l1|p1;
        assert_eq!(p1l1.e0(), 3.);
        assert_eq!(p1l1.e1(), -7.);
        assert_eq!(p1l1.e2(), 14.);
        assert_eq!(p1l1.e3(), -7.);
	}


    #[test]
    fn multivector_ip_plane_ideal_line() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);

        // a*e01 + b*e02 + c*e03
        let l1 = IdealLine::new(-2., 1., 4.);

        let p1l1:Plane = p1 | l1;
        assert_eq!(p1l1.e0(), -12.);
    }

    #[test]
    fn multivector_ip_plane_point() {
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p1 = Plane::new(1., 2., 3., 4.);
        // x*e_032 + y*e_013 + z*e_021 + e_123
        let p2 = Point::new(-2., 1., 4.);

		let p1p2:Line = p1 | p2;
        assert_eq!(p1p2.e01(), -5.);
        assert_eq!(p1p2.e02(), 10.);
        assert_eq!(p1p2.e03(), -5.);
        assert_eq!(p1p2.e12(), 3.);
        assert_eq!(p1p2.e31(), 2.);
        assert_eq!(p1p2.e23(), 1.);
    }
    #[test]
    fn multivector_ip_point_plane() {
        // x*e_032 + y*e_013 + z*e_021 + e_123
        let p1 = Point::new(-2., 1., 4.);
        // d*e_0 + a*e_1 + b*e_2 + c*e_3
        let p2 = Plane::new(1., 2., 3., 4.);

		let p1p2:Line = p1 | p2;
        assert_eq!(p1p2.e01(), -5.);
        assert_eq!(p1p2.e02(), 10.);
        assert_eq!(p1p2.e03(), -5.);
        assert_eq!(p1p2.e12(), 3.);
        assert_eq!(p1p2.e31(), 2.);
        assert_eq!(p1p2.e23(), 1.);
    }

    #[test]
    fn multivector_ip_line_line() {
        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(1., 0., 0., 3., 2., 1.);
        let l2 = Line::new(0., 1., 0., 4., 1., -2.);

        let l1l2:f32 = l1 | l2;
        assert_eq!(l1l2, -12.);
    }

    #[test]
    fn multivector_ip_line_point() {
        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(0., 0., 1., 3., 2., 1.);
        // x*e_032 + y*e_013 + z*e_021 + e_123
        let p2 = Point::new(-2., 1., 4.);

        let l1p2:Plane = l1 | p2;
        assert_eq!(l1p2.e0(), 0.);
        assert_eq!(l1p2.e1(), -3.);
        assert_eq!(l1p2.e2(), -2.);
        assert_eq!(l1p2.e3(), -1.);
    }
    #[test]
    fn multivector_ip_point_line() {
        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(0., 0., 1., 3., 2., 1.);
        // x*e_032 + y*e_013 + z*e_021 + e_123
        let p2 = Point::new(-2., 1., 4.);

        let l1p2:Plane = p2 | l1;
        assert_eq!(l1p2.e0(), 0.);
        assert_eq!(l1p2.e1(), -3.);
        assert_eq!(l1p2.e2(), -2.);
        assert_eq!(l1p2.e3(), -1.);
    }

    #[test]
    fn multivector_ip_point_point() {
        // x*e_032 + y*e_013 + z*e_021 + e_123
        let p1 = Point::new(1., 2., 3.);
        let p2 = Point::new(-2., 1., 4.);

        let p1p2:f32 = p1 | p2;
        assert_eq!(p1p2, -1.);
    }

    #[test]
    fn multivector_ip_project_point_to_line() {
        let p1 = Point::new(2., 2., 0.);
        let p2 = Point::new(0., 0., 0.);
        let p3 = Point::new(1., 0., 0.);
        let l:Line = p2 & p3;
        let mut p4 = (l | p1) ^ l;//:Point
        p4.normalize();

        approx_eq(p4.e123(), 1.);
        approx_eq(p4.x(), 2.);
        approx_eq(p4.y(), 0.);
        approx_eq(p4.z(), 0.);
    }


}