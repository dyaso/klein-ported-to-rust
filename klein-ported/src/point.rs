use super::detail::sse::{rcp_nr1};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A point is represented as the multivector
/// $x\mathbf{e}_{032} + y\mathbf{e}_{013} + z\mathbf{e}_{021} +
/// \mathbf{e}_{123}$. The point has a trivector representation because it is
/// the fixed point of 3 planar reflections (each of which is a grade-1
/// multivector). In practice, the coordinate mapping can be thought of as an
/// implementation detail.
#[derive(Copy, Clone)]
pub struct Point {
    pub p3_: __m128,
}

// impl Default for Point {
//     #[inline]
//     fn default() -> Point {
//         Point {p3_: unsafe {_mm_setzero_ps()} }
//     }
// }

// impl Default for [f32; 4] {
//     #[inline]
//     fn default() -> [f32; 4] {
//         [0.0,0.0,0.0,0.0]
//     }
// }

impl From<__m128> for Point {
    fn from(xmm: __m128) -> Point {
        Point { p3_: xmm }
    }
}

impl Point {
    /// Component-wise constructor (homogeneous coordinate is automatically
    /// initialized to 1)
    pub fn new(x: f32, y: f32, z: f32) -> Point {
        Point {
            p3_: unsafe { _mm_set_ps(z, y, x, 1.0) },
        }
    }

    pub fn x(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p3_);
        }
        return out[1];
    }

    pub fn y(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p3_);
        }
        return out[2];
    }

    pub fn z(self) -> f32 {
        let mut out = <[f32; 4]>::default();
        unsafe {
            _mm_store_ps(&mut out[0], self.p3_);
        }
        return out[3];
    }

    /// The homogeneous coordinate `w` is exactly $1$ when normalized.
    pub fn w(self) -> f32 {
        let mut out: f32 = 0.0;
        unsafe {
            _mm_store_ss(&mut out, self.p3_);
        }
        return out;
    }
}

use std::ops::Add;

impl Add for Point {
    type Output = Point;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Point::from(_mm_add_ps(self.p3_, rhs.p3_)) }
    }
}

use std::ops::Sub;

/// Point subtraction
impl Sub for Point {
    type Output = Point;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Point::from(_mm_sub_ps(self.p3_, rhs.p3_)) }
    }
}

use std::ops::Mul;
/// Point uniform scale
impl Mul<f32> for Point {
    type Output = Point;
    #[inline]
    fn mul(self, s: f32) -> Self
    {
        unsafe{let c = Point::from(_mm_mul_ps(self.p3_, _mm_set1_ps(s)));
        return c;
    }
    }
}

impl Mul<Point> for f32 {
    type Output = Point;
    #[inline]
    fn mul(self, p: Point) -> Point
    {
        return p * self;
    }
}

use std::ops::Div;
/// Point uniform inverse scale
impl Div<f32> for Point {
    type Output = Point;
    #[inline]
    fn div(self, s: f32) -> Self
    {
        unsafe{let c = Point::from(_mm_mul_ps(self.p3_, rcp_nr1(_mm_set1_ps(s))));
        return c;
    }
    }
}

use std::ops::Neg;
impl Neg for Point {
    type Output = Point;
    /// Unary minus (leaves homogeneous coordinate untouched)
    #[inline]
    fn neg(self) -> Self::Output
    {
        return Point::from(unsafe{_mm_xor_ps(self.p3_, _mm_set_ps(-0.0, -0.0, -0.0, 0.0))});
    }

}



#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use super::Point;

    #[test]
    fn multivector_sum() {
        let p1 = Point::new(1.0, 2.0, 3.0);
        let p2 = Point::new(2.0, 3.0, -1.0);
        let p3 = p1 + p2;
        approx_eq(p3.x(), 1.0 + 2.0);
        approx_eq(p3.y(), 2.0 + 3.0);
        approx_eq(p3.z(), 3.0 + -1.0);

        let p4 = p1 - p2;
        approx_eq(p4.x(), 1.0 - 2.0);
        approx_eq(p4.y(), 2.0 - 3.0);
        approx_eq(p4.z(), 3.0 - -1.0);

        // // Adding rvalue to lvalue
        let p5 = Point::new(1.0, 2.0, 3.0) + p2;
        assert_eq!(p5.x(), 1.0 + 2.0);
        assert_eq!(p5.y(), 2.0 + 3.0);
        assert_eq!(p5.z(), 3.0 + -1.0);

        // // Adding rvalue to rvalue
        let p6 = Point::new(1.0, 2.0, 3.0) + Point::new(2.0, 3.0, -1.0);
        assert_eq!(p6.x(), 1.0 + 2.0);
        assert_eq!(p6.y(), 2.0 + 3.0);
        assert_eq!(p6.z(), 3.0 + -1.0);
    }
}
