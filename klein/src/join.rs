#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{Point, Plane};

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
    fn not(self) -> Self::Output
    {
    	Plane::from(self.p3_)
    }
}

impl Not for Plane {
    type Output = Point;
    #[inline]
    fn not(self) -> Self::Output
    {
    	Point::from(self.p0_)
    }
}
