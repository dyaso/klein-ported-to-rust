#![cfg(target_arch = "x86_64")]

use crate::{Branch, Dual, IdealLine, Line, Plane, Point};

/// Poincaré Dual
///
/// The Poincaré Dual of an element is the "subspace complement" of the
/// argument with respect to the pseudoscalar in the exterior algebra. In
/// practice, it is a relabeling of the coordinates to their
/// dual-coordinates and is used most often to implement a "join" operation
/// in terms of the exterior product of the duals of each operand.
///
/// Ex: The dual of the point $\mathbf{e}_{123} + 3\mathbf{e}_{013} -
/// 2\mathbf{e}_{021}$ (the point at
/// $(0, 3, -2)$) is the plane
/// $\mathbf{e}_0 + 3\mathbf{e}_2 - 2\mathbf{e}_3$.
use std::ops::Not;
impl Not for Point {
    type Output = Plane;
    #[inline]
    fn not(self) -> Self::Output {
        Plane::from(self.p3_)
    }
}

impl Not for Plane {
    type Output = Point;
    #[inline]
    fn not(self) -> Self::Output {
        Point::from(self.p0_)
    }
}

impl Not for Line {
    type Output = Line;
    #[inline]
    fn not(self) -> Self::Output {
        Line::from(self.p2_, self.p1_)
    }
}

impl Not for IdealLine {
    type Output = Branch;
    #[inline]
    fn not(self) -> Self::Output {
        Branch::from(self.p2_)
    }
}

impl Not for Branch {
    type Output = IdealLine;
    #[inline]
    fn not(self) -> Self::Output {
        IdealLine::from(self.p1_)
    }
}

impl Not for Dual {
    type Output = Dual;
    #[inline]
    fn not(self) -> Self::Output {
        Dual {
            p: self.q,
            q: self.p,
        }
    }
}

use std::ops::BitAnd;

impl BitAnd<Point> for Point {
    type Output = Line;
    #[inline]
    fn bitand(self, rhs: Point) -> Self::Output {
        return !(!self ^ !rhs);
    }
}

impl BitAnd<Line> for Point {
    type Output = Plane;
    #[inline]
    fn bitand(self, rhs: Line) -> Self::Output {
        return !(!self ^ !rhs);
    }
}

//
impl BitAnd<Point> for Line {
    type Output = Plane;
    #[inline]
    fn bitand(self, rhs: Point) -> Self::Output {
        return rhs & self;
    }
}

impl BitAnd<Branch> for Point {
    type Output = Plane;
    #[inline]
    fn bitand(self, rhs: Branch) -> Self::Output {
        return !(!self ^ !rhs);
    }
}

impl BitAnd<Point> for Branch {
    type Output = Plane;
    #[inline]
    fn bitand(self, rhs: Point) -> Self::Output {
        return rhs & self;
    }
}

impl BitAnd<IdealLine> for Point {
    type Output = Plane;
    #[inline]
    fn bitand(self, rhs: IdealLine) -> Self::Output {
        return !(!self ^ !rhs);
    }
}

impl BitAnd<Point> for IdealLine {
    type Output = Plane;
    #[inline]
    fn bitand(self, rhs: Point) -> Self::Output {
        return rhs & self;
    }
}

impl BitAnd<Point> for Plane {
    type Output = Dual;
    #[inline]
    fn bitand(self, rhs: Point) -> Self::Output {
        return !(!self ^ !rhs);
    }
}

impl BitAnd<Plane> for Point {
    type Output = Dual;
    #[inline]
    fn bitand(self, rhs: Plane) -> Self::Output {
        return !(!self ^ !rhs);
    }
}

#[cfg(test)]
mod tests {
    #![cfg(target_arch = "x86_64")]

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use crate::{Line, Plane, Point};

    #[test]
    fn multivector_rp_pos_z_line() {
        let p1 = Point::new(0., 0., 0.);
        let p2 = Point::new(0., 0., 1.);
        let p12: Line = p1 & p2;
        assert_eq!(p12.e12(), 1.);
    }

    #[test]
    fn multivector_rp_pos_y_line() {
        let p1 = Point::new(0., -1., 0.);
        let p2 = Point::new(0., 0., 0.);
        let p12: Line = p1 & p2;
        assert_eq!(p12.e31(), 1.);
    }

    #[test]
    fn multivector_rp_pos_x_line() {
        let p1 = Point::new(-2., 0., 0.);
        let p2 = Point::new(-1., 0., 0.);
        let p12: Line = p1 & p2;
        assert_eq!(p12.e23(), 1.);
    }

    #[test]
    fn multivector_rp_plane_construction() {
        let p1 = Point::new(1., 3., 2.);
        let p2 = Point::new(-1., 5., 2.);
        let p3 = Point::new(2., -1., -4.);

        let p123: Plane = p1 & p2 & p3;

        // Check that all 3 points lie on the plane
        assert_eq!(p123.e1() + p123.e2() * 3. + p123.e3() * 2. + p123.e0(), 0.);
        assert_eq!(-p123.e1() + p123.e2() * 5. + p123.e3() * 2. + p123.e0(), 0.);
        assert_eq!(p123.e1() * 2. - p123.e2() - p123.e3() * 4. + p123.e0(), 0.);
    }

    // from file: test/test_metric.cpp

    #[test]
    fn measure_point_to_point() {
        let p1 = Point::new(1., 0., 0.);
        let p2 = Point::new(0., 1., 0.);
        let l: Line = p1 & p2;
        // Produce the squared distance between p1 and p2
        assert_eq!(l.squared_norm(), 2.);
    }

    #[test]
    fn measure_point_to_plane() {
        //    Plane p2
        //    /
        //   / \ line perpendicular to
        //  /   \ p2 through p1
        // 0------x--------->
        //        p1

        // (2, 0, 0)
        let p1 = Point::new(2., 0., 0.);
        // Plane x - y = 0
        let mut p2 = Plane::new(1., -1., 0., 0.);
        p2.normalize();
        // Distance from point p1 to plane p2
        let root_two = f32::sqrt(2.);
        println!("p2 {} {}", (p1 & p2).scalar(), root_two);

        approx_eq(f32::abs((p1 & p2).scalar()), root_two);
        // approx_eq(f32::abs((p1 ^ p2).e0123()), root_two);
    }

    #[test]
    fn measure_point_to_line() {
        let l = Line::new(0., 1., 0., 1., 0., 0.);
        let p = Point::new(0., 1., 2.);
        let distance = (l & p).norm();
        approx_eq(distance, f32::sqrt(2.));
    }
}
