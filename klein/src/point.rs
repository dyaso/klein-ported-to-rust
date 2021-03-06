#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{rcp_nr1};

pub type Element = __m128;

/// A point is represented as the multivector
/// $x\mathbf{e}_{032} + y\mathbf{e}_{013} + z\mathbf{e}_{021} +
/// \mathbf{e}_{123}$. The point has a trivector representation because it is
/// the fixed point of 3 planar reflections (each of which is a grade-1
/// multivector). In practice, the coordinate mapping can be thought of as an
/// implementation detail.
#[derive(Copy, Clone, Debug)]
pub struct Point {
    pub p3_: __m128,
}

// impl From<__m128> for Point {
//     fn from(xmm: __m128) -> Self {
//         unsafe {
//             Motor {
//                 p3_: xmm,
//             }
//         }
//     }
// }

pub fn origin() -> Point {
    Point::new(0., 0., 0.)
}

// // trait AsRef<T: ?Sized> {
// //     fn as_ref(&self) -> &T;
// // }
// impl AsRef<Point> for Point {
//     fn as_ref(&self) -> &Point {
//         &(&self)
//     }
// }

use std::fmt;

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "\nPlane\te032\te013\te21\te123\n\t{:.3}\t{:.3}\t{:.3}\t{:.3}\n",
            self.x(),
            self.y(),
            self.z(),
            self.w()
        )
    }
}

// impl Default for [f32; 4] {
//     #[inline]
//     fn default() -> [f32; 4] {
//         [0.0,0.0,0.0,0.0]
//     }
// }

impl Point {
    pub fn scale(&mut self, scale: f32) {
        unsafe {
            self.p3_ = _mm_mul_ss(
                _mm_mul_ps(self.p3_, _mm_set1_ps(scale)),
                _mm_set_ss(1. / scale),
            );
        }
    }

    pub fn scaled(self, scale: f32) -> Point {
        let mut out = Point::from(self.p3_);
        out.scale(scale);
        out
    }

    /// Create a normalized direction
    pub fn direction(x: f32, y: f32, z: f32) -> Point {
        unsafe {
            let mut p = Point {
                p3_: _mm_set_ps(z, y, x, 0.),
            };
            p.normalize();
            p
        }
    }

    /// Component-wise constructor (homogeneous coordinate is automatically
    /// initialized to 1)
    pub fn new(x: f32, y: f32, z: f32) -> Point {
        Point {
            p3_: unsafe { _mm_set_ps(z, y, x, 1.0) },
        }
    }

    /// Normalize this point (division is done via rcpps with an additional
    /// Newton-Raphson refinement).
    pub fn normalize(&mut self) {
        unsafe {
            // if self.w() == 0. {
            //     // it's a direction, a point at infinity
            //     let tmp = rsqrt_nr1(hi_dp_bc(self.p3_, self.p3_));
            //     self.p3_ = _mm_mul_ps(self.p3_, tmp);
            // } else {
            //     // it's a regular point
            let tmp = rcp_nr1(_mm_shuffle_ps(self.p3_, self.p3_, 0));
            self.p3_ = _mm_mul_ps(self.p3_, tmp);
            // }
        }
    }

    pub fn invert(&mut self) {
        unsafe {
            let inv_norm = rcp_nr1(_mm_shuffle_ps(self.p3_, self.p3_, 0));
            self.p3_ = _mm_mul_ps(inv_norm, self.p3_);
            self.p3_ = _mm_mul_ps(inv_norm, self.p3_);
        }
    }

    pub fn inverse(self) -> Point {
        let mut out = Point::clone(&self);
        out.invert();
        out
    }

    get_basis_blade_fn!(e032, e023, p3_, 1);
    get_basis_blade_fn!(e013, e031, p3_, 2);
    get_basis_blade_fn!(e021, e012, p3_, 3);

    pub fn x(self) -> f32 {
        self.e032()
    }

    pub fn y(self) -> f32 {
        self.e013()
    }

    pub fn z(self) -> f32 {
        self.e021()
    }

    /// The homogeneous coordinate `w` is exactly $1$ when normalized.
    pub fn w(self) -> f32 {
        let mut out: f32 = 0.0;
        unsafe {
            _mm_store_ss(&mut out, self.p3_);
        }
        out
    }

    pub fn e123(self) -> f32 {
        self.w()
    }
}

common_operations!(Point, p3_);
/// Point uniform scale

macro_rules! mul_scalar_by_point {
    ($s:ty) => {
        impl Mul<Point> for $s {
            type Output = Point;
            #[inline]
            fn mul(self, l: Point) -> Point {
                l * (self as f32)
            }
        }
    };
}

mul_scalar_by_point!(f32);
mul_scalar_by_point!(f64);
mul_scalar_by_point!(i32);

/// Point uniform inverse scale
impl<T: Into<f32>> Div<T> for Point {
    type Output = Point;
    #[inline]
    fn div(self, s: T) -> Self {
        unsafe { Point::from(_mm_mul_ps(self.p3_, rcp_nr1(_mm_set1_ps(s.into())))) }
    }
}

use std::ops::Neg;
impl Neg for Point {
    type Output = Point;
    /// Unary minus (leaves homogeneous coordinate untouched)
    #[inline]
    fn neg(self) -> Self::Output {
        Point::from(unsafe { _mm_xor_ps(self.p3_, _mm_set_ps(-0.0, -0.0, -0.0, 0.0)) })
    }
}

#[cfg(test)]
mod tests {
    #![cfg(target_arch = "x86_64")]

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
