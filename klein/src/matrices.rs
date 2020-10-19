#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{ApplyTo, Motor};

/// 4x4 column-major matrix (used for converting rotors/motors to matrix form to
/// upload to shaders).
#[derive(Debug)]
pub enum Mat4x4 {
    Cols([__m128; 4]),
    Data([f32; 16]),
}

impl Default for Mat4x4 {
    fn default() -> Mat4x4 {
        unsafe {
            Mat4x4::Cols {
                0: [
                    _mm_setzero_ps(),
                    _mm_setzero_ps(),
                    _mm_setzero_ps(),
                    _mm_setzero_ps(),
                ],
            }
        }
    }
}

pub enum Mat3x4 {
    Cols([__m128; 4]),
    Data([f32; 16]),
}

impl Default for Mat3x4 {
    fn default() -> Mat3x4 {
        unsafe {
            Mat3x4::Cols {
                0: [
                    _mm_setzero_ps(),
                    _mm_setzero_ps(),
                    _mm_setzero_ps(),
                    _mm_setzero_ps(),
                ],
            }
        }
    }
}

macro_rules! swizzle {
    ($id:ident, $a:expr, $b:expr, $c:expr, $d:expr) => {
        _mm_shuffle_ps($id, $id, ($a << 6) | ($b << 4) | ($c << 2) | $d)
    };
}

impl ApplyTo<__m128> for Mat4x4 {
    fn apply_to(self, xyzw: __m128) -> __m128 {
        match self {
            Mat4x4::Cols(cols) => unsafe {
                let mut out = _mm_mul_ps(cols[0], swizzle!(xyzw, 0, 0, 0, 0));
                out = _mm_add_ps(out, _mm_mul_ps(cols[1], swizzle!(xyzw, 1, 1, 1, 1)));
                out = _mm_add_ps(out, _mm_mul_ps(cols[2], swizzle!(xyzw, 2, 2, 2, 2)));
                out = _mm_add_ps(out, _mm_mul_ps(cols[3], swizzle!(xyzw, 3, 3, 3, 3)));
                out
            },
            _ => unreachable!(),
        }
    }
}

impl ApplyTo<__m128> for Mat3x4 {
    fn apply_to(self, xyzw: __m128) -> __m128 {
        match self {
            Mat3x4::Cols(cols) => unsafe {
                let mut out = _mm_mul_ps(cols[0], swizzle!(xyzw, 0, 0, 0, 0));
                out = _mm_add_ps(out, _mm_mul_ps(cols[1], swizzle!(xyzw, 1, 1, 1, 1)));
                out = _mm_add_ps(out, _mm_mul_ps(cols[2], swizzle!(xyzw, 2, 2, 2, 2)));
                out = _mm_add_ps(out, _mm_mul_ps(cols[3], swizzle!(xyzw, 3, 3, 3, 3)));
                out
            },
            _ => unreachable!(),
        }
    }
}

use crate::detail::matrix::mat4x4_12;

impl Motor {
    /// Convert this motor to a 3x4 column-major matrix representing this
    /// motor's action as a linear transformation. The motor must be normalized
    /// for this conversion to produce well-defined results, but is more
    /// efficient than a 4x4 matrix conversion.
    pub fn as_mat3x4(self) -> Mat3x4 {
        let out = Mat3x4::default();
        match out {
            Mat3x4::Cols(mut cols) => {
                mat4x4_12(true, true, self.p1_, &self.p2_, &mut cols);
                Mat3x4::Cols(cols)
            }
            _ => unreachable!(),
        }
    }

    /// Convert this motor to a 4x4 column-major matrix representing this
    /// motor's action as a linear transformation.
    pub fn as_mat4x4(self) -> Mat4x4 {
        let out = Mat4x4::default();
        match out {
            Mat4x4::Cols(mut cols) => {
                mat4x4_12(true, false, self.p1_, &self.p2_, &mut cols);
                Mat4x4::Cols(cols)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use crate::{ApplyTo, Mat3x4, Mat4x4, Motor};

    #[test]
    fn motor_to_matrix() {
        let mut buf = <[f32; 4]>::default();
        let m = Motor::new(1., 4., 3., 2., 5., 6., 7., 8.);

        unsafe {
            let p1: __m128 = _mm_set_ps(1., 2., 1., -1.);
            let m_mat: Mat4x4 = m.as_mat4x4();
            let p2: __m128 = m_mat.apply_to(p1);
            _mm_storeu_ps(&mut buf[0], p2);
        }

        assert_eq!(buf[0], -12.);
        assert_eq!(buf[1], -86.);
        assert_eq!(buf[2], -86.);
        assert_eq!(buf[3], 30.);
    }

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    #[test]
    fn motor_to_matrix_3x4() {
        let mut m = Motor::new(1., 4., 3., 2., 5., 6., 7., 8.);
        m.normalize();
        let mut buf = <[f32; 4]>::default();

        unsafe {
            let p1 = _mm_set_ps(1., 2., 1., -1.);
            let m_mat: Mat3x4 = m.as_mat3x4();
            let p2 = m_mat.apply_to(p1);
            _mm_storeu_ps(&mut buf[0], p2);
        }

        approx_eq(buf[0], -12. / 30.);
        approx_eq(buf[1], -86. / 30.);
        approx_eq(buf[2], -86. / 30.);
        assert_eq!(buf[3], 1.);
    }
}
