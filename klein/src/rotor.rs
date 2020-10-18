#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sandwich::{sw012, sw_mm_three, sw_mm_two};
use crate::detail::sse::{dp_bc, rsqrt_nr1}; //rcp_nr1, hi_dp, hi_dp_bc, };

use crate::util::ApplyOp;
use crate::{Branch, Line, Point};

#[derive(Default)]
pub struct EulerAngles {
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl EulerAngles {
    pub fn new(roll: f32, pitch: f32, yaw: f32) -> EulerAngles {
        EulerAngles {
            roll: roll,
            pitch: pitch,
            yaw: yaw,
        }
    }
}

/// \defgroup rotor Rotors
///
/// The rotor is an entity that represents a rigid rotation about an axis.
/// To apply the rotor to a supported entity, the call operator is available.
///
/// !!! example
///
/// ```c++
///     // Initialize a point at (1, 3, 2)
///     kln::point p{1.f, 3.f, 2.f};
///        
///     // Create a normalized rotor representing a pi/2 radian
///     // rotation about the xz-axis.
///     kln::rotor r{kln::pi * 0.5f, 1.f, 0.f, 1.f};
///     
///     // Rotate our point using the created rotor
///     kln::point rotated = r(p);
/// ```
/// We can rotate lines and planes as well using the rotor's call operator.
///
/// Rotors can be multiplied to one another with the `*` operator to create
/// a new rotor equivalent to the application of each factor.
///
/// !!! example
///
/// ```c++
///     // Create a normalized rotor representing a $\frac{\pi}{2}$ radian
///     // rotation about the xz-axis.
///     kln::rotor r1{kln::pi * 0.5f, 1.f, 0.f, 1.f};
///        
///     // Create a second rotor representing a $\frac{\pi}{3}$ radian
///     // rotation about the yz-axis.
///     kln::rotor r2{kln::pi / 3.f, 0.f, 1.f, 1.f};
///        
///     // Use the geometric product to create a rotor equivalent to first
///     // applying r1, then applying r2. Note that the order of the
///     // operands here is significant.
///     kln::rotor r3 = r2 * r1;
/// ```
///
/// The same `*` operator can be used to compose the rotor's action with other
/// translators and motors.

pub type Rotor = Branch;

impl Rotor {
    /// Convenience constructor. Computes transcendentals and normalizes
    /// rotation axis.
    pub fn rotor(ang_rad: f32, x: f32, y: f32, z: f32) -> Rotor {
        let norm: f32 = f32::sqrt(x * x + y * y + z * z);
        let inv_norm: f32 = 1. / norm;

        let half: f32 = 0.5 * ang_rad;
        // Rely on compiler to coalesce these two assignments into a single
        // sincos call at instruction selection time
        let sin_ang = f32::sin(half);
        let scale = sin_ang * inv_norm;
        unsafe {
            let mut p1_ = _mm_set_ps(z, y, x, f32::cos(half));
            p1_ = _mm_mul_ps(p1_, _mm_set_ps(scale, scale, scale, 1.));
            return Rotor::from(p1_)
        }
    }

    /// Fast load operation for packed data that is already normalized. The
    /// argument `data` should point to a set of 4 float values with layout `(a,
    /// b, c, d)` corresponding to the multivector
    /// $a + b\mathbf{e}_{23} + c\mathbf{e}_{31} + d\mathbf{e}_{12}$.
    ///
    /// !!! danger
    ///
    /// The rotor data loaded this way *must* be normalized. That is, the
    /// rotor $r$ must satisfy $r\widetilde{r} = 1$.
    pub fn load_normalized(&mut self, data: &f32) {
        unsafe {
            self.p1_ = _mm_loadu_ps(data);
        }
    }

    pub fn from_euler_angles(ea: &EulerAngles) -> Rotor {
        // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#cite_note-3
        let half_yaw: f32 = ea.yaw * 0.5;
        let half_pitch: f32 = ea.pitch * 0.5;
        let half_roll: f32 = ea.roll * 0.5;
        let cos_y: f32 = f32::cos(half_yaw);
        let sin_y: f32 = f32::sin(half_yaw);
        let cos_p: f32 = f32::cos(half_pitch);
        let sin_p: f32 = f32::sin(half_pitch);
        let cos_r: f32 = f32::cos(half_roll);
        let sin_r: f32 = f32::sin(half_roll);

        unsafe {
            let p1_ = _mm_set_ps(
                cos_r * cos_p * sin_y - sin_r * sin_p * cos_y,
                cos_r * sin_p * cos_y + sin_r * cos_p * sin_y,
                sin_r * cos_p * cos_y - cos_r * sin_p * sin_y,
                cos_r * cos_p * cos_y + sin_r * sin_p * sin_y,
            );

            return Rotor::from(p1_).normalized_rotor();
        }
    }

    /// Constrains the rotor to traverse the shortest arc
    pub fn constrain(&mut self) {
        unsafe {
            let um = _mm_and_ps(self.p1_, _mm_set_ss(-0.));
            let mask: __m128 = _mm_shuffle_ps(um, um, 0);
            self.p1_ = _mm_xor_ps(mask, self.p1_);
        }
    }

    pub fn constrained(self) -> Rotor {
        let mut out = Rotor::clone(&self);
        out.constrain();
        return out;
    }

    pub fn as_euler_angles(self) -> EulerAngles {
        let pi = std::f32::consts::PI;
        let pi_2 = pi / 2.;

        let mut ea = EulerAngles::default();


        #[repr(align(16))]
        struct ArrayAlignedTo16ByteBoundary {mem: [f32; 4] };

        let mut buf = ArrayAlignedTo16ByteBoundary{mem: <[f32; 4]>::default()};
        self.store(&mut buf.mem[0]);
        let test = buf.mem[1] * buf.mem[2] + buf.mem[3] * buf.mem[0];

        if test > 0.4999 {
            ea.roll = 2. * f32::atan2(buf.mem[1], buf.mem[0]);
            ea.pitch = pi_2;
            ea.yaw = 0.;
            return ea;
        } else if test < -0.4999 {
            ea.roll = -2. * f32::atan2(buf.mem[1], buf.mem[0]);
            ea.pitch = -pi_2;
            ea.yaw = 0.;
            return ea;
        }

        let buf1_2 = buf.mem[1] * buf.mem[1];
        let buf2_2 = buf.mem[2] * buf.mem[2];
        let buf3_2 = buf.mem[3] * buf.mem[3];

        ea.roll = f32::atan2(
            2. * (buf.mem[0] * buf.mem[1] + buf.mem[2] * buf.mem[3]),
            1. - 2. * (buf1_2 + buf2_2),
        );

        let sinp = 2. * (buf.mem[0] * buf.mem[2] - buf.mem[1] * buf.mem[3]);
        if f32::abs(sinp) > 1. {
            ea.pitch = f32::copysign(pi_2, sinp);
        } else {
            ea.pitch = f32::asin(sinp);
        }

        ea.yaw = f32::atan2(
            2. * (buf.mem[0] * buf.mem[3] + buf.mem[1] * buf.mem[2]),
            1. - 2. * (buf2_2 + buf3_2),
        );
        return ea;
    }

    pub fn normalize_rotor(&mut self) {
        unsafe {
            let inv_norm: __m128 = rsqrt_nr1(dp_bc(self.p1_, self.p1_));
            self.p1_ = _mm_mul_ps(self.p1_, inv_norm);
        }
    }

    pub fn normalized_rotor(self) -> Self {
        let mut out = Self::from(self.p1_);
        out.normalize_rotor();
        return out;
    }
}

impl PartialEq for Rotor {
    fn eq(&self, other: &Rotor) -> bool {
        unsafe { return _mm_movemask_ps(_mm_cmpeq_ps(self.p1_, other.p1_)) == 0b1111 }
    }
}


impl ApplyOp<Branch> for Rotor {
    fn apply_to(self, rhs: Branch) -> Branch {
        Branch{p1_: sw_mm_two(rhs.p1_, self.p1_)}
    }
}

//        [[nodiscard]] line KLN_VEC_CALL operator()(line const& l) const noexcept
impl ApplyOp<Line> for Rotor {
    /// Conjugates a line $\ell$ with this rotor and returns the result
    /// $r\ell \widetilde{r}$.
    fn apply_to(self, l: Line) -> Line {
        let (branch, ideal) = sw_mm_three(l.p1_, l.p2_, self.p1_);
        return Line::from(branch, ideal);
    }
}

impl ApplyOp<Point> for Rotor {
    /// Conjugates a point $p$ with this rotor and returns the result
    /// $rp\widetilde{r}$.
    fn apply_to(self, p: Point) -> Point {
        // NOTE: Conjugation of a plane and point with a rotor is identical
        unsafe { return Point::from(sw012(false, p.p3_, self.p1_, _mm_setzero_ps())) }
    }
}

// impl From<&EulerAngles> for Rotor {
// }

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use crate::{ApplyOp, EulerAngles, Line, Point, Rotor};

    #[test]
    fn rotor_line() {
        // Make an unnormalized rotor to verify correctness
        let data: [f32; 4] = [1., 4., -3., 2.];
        let mut r = Rotor::default();
        r.load_normalized(&data[0]);
        // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
        let l1 = Line::new(-1., 2., -3., -6., 5., 4.);
        let l2 = r.apply_to(l1);
        assert_eq!(l2.e01(), -110.);
        assert_eq!(l2.e02(), 20.);
        assert_eq!(l2.e03(), 10.);
        assert_eq!(l2.e12(), -240.);
        assert_eq!(l2.e31(), 102.);
        assert_eq!(l2.e23(), -36.);
    }

    #[test]
    fn rotor_point() {
        let pi = std::f32::consts::PI;
        let r = Rotor::rotor(pi * 0.5, 0., 0., 1.);
        let p1 = Point::new(1., 0., 0.);
        let p2: Point = r.apply_to(p1);
        assert_eq!(p2.x(), 0.);
        approx_eq(p2.y(), -1.); //approx
        assert_eq!(p2.z(), 0.);
    }

    #[test]
    fn euler_angles_precision() {
        let pi = std::f32::consts::PI;
        let ea1 = EulerAngles::new(pi * 0.2, pi * 0.2, 0.);
        let r1 = Rotor::from_euler_angles(&ea1);
        let ea2 = r1.as_euler_angles();

        approx_eq(ea1.roll, ea2.roll);
        approx_eq(ea1.pitch, ea2.pitch);
        approx_eq(ea1.yaw, ea2.yaw);
    }

    #[test]
    fn euler_angles() {
        // Make 3 rotors about the x, y, and z-axes.
        let rx = Rotor::rotor(1., 1., 0., 0.);
        let ry = Rotor::rotor(1., 0., 1., 0.);
        let rz = Rotor::rotor(1., 0., 0., 1.);

        let r: Rotor = rx * ry * rz;
        let ea = r.as_euler_angles();
        approx_eq(ea.roll, 1.);
        approx_eq(ea.pitch, 1.);
        approx_eq(ea.yaw, 1.);

        let r2 = Rotor::from_euler_angles(&ea);

        #[repr(align(16))]
        struct ArrayAlignedTo16ByteBoundary {mem: [f32; 8] };

        let mut buf = ArrayAlignedTo16ByteBoundary{mem: <[f32; 8]>::default()};
        //let mut buf = <[f32; 8]>::default();

        r.store(&mut buf.mem[0]);
        r2.store(&mut buf.mem[4]);
        for i in 0..4  {
            approx_eq(buf.mem[i], buf.mem[i+4]);
        }
    }

    #[test]
    fn rotor_constrain() {
        let mut r1 = Rotor::rotor(1., 2., 3., 4.);
        let mut r2 = r1.constrained();
        assert_eq!(r1, r2);

        r1 = -r1;
        r2 = r1.constrained();
        assert_eq!(r1, -r2);
    }

    #[test]
    fn rotor_sqrt() {
        let pi = std::f32::consts::PI;
        let r = Rotor::rotor(pi * 0.5, 1., 2., 3.);

        let mut r2: Rotor = r.sqrt();
        r2 = r2 * r2;
        approx_eq(r2.scalar(), r.scalar());
        approx_eq(r2.e23(), r.e23());
        approx_eq(r2.e31(), r.e31());
        approx_eq(r2.e12(), r.e12());
    }

    #[test]
    fn normalize_rotor() {
        unsafe {
            let mut r = Rotor::from(_mm_set_ps(4., -3., 3., 28.));
            //	    r.p1_ = _mm_set_ps(4.f, -3.f, 3.f, 28.f);
            r.normalize();
            let norm: Rotor = r * r.reverse();
            approx_eq(norm.scalar(), 1.);
            approx_eq(norm.e12(), 0.);
            approx_eq(norm.e31(), 0.);
            approx_eq(norm.e23(), 0.);
        }
    }
}
