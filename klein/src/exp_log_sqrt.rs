#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{hi_dp, rcp_nr1, sqrt_nr1};

use crate::detail::exp_log::{simd_exp, simd_log};
// use crate::detail::geometric_product::{gp_dl};

use crate::{Branch, IdealLine, Line, Motor, Rotor, Translator};

/// \defgroup exp_log Exponential and Logarithm
/// @{
///
/// The group of rotations, translations, and screws (combined rotatation and
/// translation) is _nonlinear_. This means, given say, a rotor $\mathbf{r}$,
/// the rotor
/// $\frac{\mathbf{r}}{2}$ _does not_ correspond to half the rotation.
/// Similarly, for a motor $\mathbf{m}$, the motor $n \mathbf{m}$ is not $n$
/// applications of the motor $\mathbf{m}$. One way we could achieve this is
/// through exponentiation; for example, the motor $\mathbf{m}^3$ will perform
/// the screw action of $\mathbf{m}$ three times. However, repeated
/// multiplication in this fashion lacks both efficiency and numerical
/// stability.
///
/// The solution is to take the logarithm of the action which maps the action to
/// a linear space. Using `log(A)` where `A` is one of `rotor`,
/// `translator`, or `motor`, we can apply linear scaling to `log(A)`,
/// and then re-exponentiate the result. Using this technique, `exp(n * log(A))`
/// is equivalent to $\mathbf{A}^n$.


pub trait Log<O> {
    fn log(self) -> O;
}

pub fn log<I: Log<O>, O>(input:I) -> O {
    input.log()
}

/// Takes the principal branch of the logarithm of the motor, returning a
/// bivector. Exponentiation of that bivector without any changes produces
/// this motor again. Scaling that bivector by $\frac{1}{n}$,
/// re-exponentiating, and taking the result to the $n$th power will also
/// produce this motor again. The logarithm presumes that the motor is
/// normalized.
impl Log<Line> for Motor {
    #[inline]
    fn log(self) -> Line {
        let mut out = Line::default();
        simd_log(self.p1_, self.p2_, &mut out.p1_, &mut out.p2_);
        out
    }
}

pub trait Exp<O> {
    fn exp(self) -> O;
}

pub fn exp<I: Exp<O>, O>(input:I) -> O {
    input.exp()
}
/// Exponentiate a line to produce a motor that posesses this line
/// as its axis. This routine will be used most often when this line is
/// produced as the logarithm of an existing rotor, then scaled to subdivide
/// or accelerate the motor's action. The line need not be a _simple bivector_
/// for the operation to be well-defined.
//impl Line {
impl Exp<Motor> for Line {
    #[inline]
    fn exp(self) -> Motor {
        let mut out = Motor::default();
        simd_exp(self.p1_, self.p2_, &mut out.p1_, &mut out.p2_);
        out
    }
}

/// Compute the logarithm of the translator, producing an ideal line axis.
/// In practice, the logarithm of a translator is simply the ideal partition
/// (without the scalar $1$).
impl Log<IdealLine> for Translator {
    #[inline]
    fn log(self) -> IdealLine {
        let mut out = IdealLine::default();
        out.p2_ = self.p2_;
        out
    }
}

/// Exponentiate an ideal line to produce a translation.
///
/// The exponential of an ideal line
/// $a \mathbf{e}_{01} + b\mathbf{e}_{02} + c\mathbf{e}_{03}$ is given as:
///
/// $$\exp{\left[a\ee_{01} + b\ee_{02} + c\ee_{03}\right]} = 1 +\
/// a\ee_{01} + b\ee_{02} + c\ee_{03}$$
impl Exp<Translator> for IdealLine {
    #[inline]
    fn exp(self) -> Translator {
        let mut out = Translator::default();
        out.p2_ = self.p2_;
        out
    }
}

/// Returns the principal branch of this rotor's logarithm. Invoking
/// `exp` on the returned `kln::branch` maps back to this rotor.
///
/// Given a rotor $\cos\alpha + \sin\alpha\left[a\ee_{23} + b\ee_{31} +\
/// c\ee_{23}\right]$, the log is computed as simply
/// $\alpha\left[a\ee_{23} + b\ee_{31} + c\ee_{23}\right]$.
/// This map is only well-defined if the
/// rotor is normalized such that $a^2 + b^2 + c^2 = 1$.
impl Log<Branch> for Rotor {
    #[inline]
    fn log(self) -> Branch {
        let mut cos_ang: f32 = 0.;
        unsafe {
            _mm_store_ss(&mut cos_ang, self.p1_);
            let ang: f32 = f32::acos(cos_ang);
            let sin_ang: f32 = f32::sin(ang);

            let mut out = Branch::default();
            out.p1_ = _mm_mul_ps(self.p1_, rcp_nr1(_mm_set1_ps(sin_ang)));
            out.p1_ = _mm_mul_ps(out.p1_, _mm_set1_ps(ang));

            if is_x86_feature_detected!("sse4.1") {
                out.p1_ = _mm_blend_ps(out.p1_, _mm_setzero_ps(), 1);
            } else {
                out.p1_ = _mm_and_ps(out.p1_, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)));
            }
            out
        }
    }
}

/// Exponentiate a branch to produce a rotor.
impl Exp<Rotor> for Branch {
    #[inline]
    fn exp(self) -> Rotor {
        // Compute the rotor angle
        let mut out = Rotor::default();
        let mut ang: f32 = 0.;
        unsafe {
            _mm_store_ss(&mut ang, sqrt_nr1(hi_dp(self.p1_, self.p1_)));
            let cos_ang = f32::cos(ang);
            let sin_ang = f32::sin(ang) / ang;

            out.p1_ = _mm_mul_ps(_mm_set1_ps(sin_ang), self.p1_);
            out.p1_ = _mm_add_ps(out.p1_, _mm_set_ps(0., 0., 0., cos_ang));
        }
        out
    }
}

/// Compute the square root of the provided rotor $r$.
impl Sqrt<Rotor> for Rotor {
    #[inline]
    fn sqrt(self) -> Rotor {
        unsafe { Rotor::from(_mm_add_ss(self.p1_, _mm_set_ss(1.))).normalized() }
    }
}

/// Compute the square root of the provided translator $t$.
impl Sqrt<Translator> for Translator {
    #[inline]
    fn sqrt(self) -> Translator {
        //let t = self;
        // t *= 0.5;
        // t
        self * 0.5
    }
}

pub trait Sqrt<O> {
    fn sqrt(self) -> O;
}

pub fn sqrt<I: Sqrt<O>, O>(input:I) -> O {
    input.sqrt()
}

/// Compute the square root of the provided motor $m$.
impl Sqrt<Motor> for Motor {
// impl Motor {
    #[inline]
    fn sqrt(self) -> Motor {
        unsafe {
            let m =
                Motor::from_rotor_and_translator(_mm_add_ss(self.p1_, _mm_set_ss(1.)), self.p2_);
            m.normalized_motor()
        }
    }
}

#[cfg(test)]
mod tests {
    #![cfg(target_arch = "x86_64")]

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    use crate::{Branch, Line, Motor, Rotor, Translator, sqrt, log, exp};

    #[test]
    fn rotor_exp_log() {
        let pi = std::f32::consts::PI;
        let r = Rotor::new(pi * 0.5, 0.3, -3., 1.);
        let b: Branch = log(r);
        let r2: Rotor = exp(b);

        approx_eq(r2.scalar(), r.scalar());
        approx_eq(r2.e12(), r.e12());
        approx_eq(r2.e31(), r.e31());
        approx_eq(r2.e23(), r.e23());
    }

    #[test]
    fn rotor_sqrt() {
        let pi = std::f32::consts::PI;
        let r1: Rotor = Rotor::new(pi * 0.5, 0.3, -3., 1.);
        let r2: Rotor = sqrt(r1);
        let r3: Rotor = r2 * r2;
        approx_eq(r1.scalar(), r3.scalar());
        approx_eq(r1.e12(), r3.e12());
        approx_eq(r1.e31(), r3.e31());
        approx_eq(r1.e23(), r3.e23());
    }

    #[test]
    fn motor_exp_log_sqrt() {
        // Construct a motor from a translator and rotor
        let pi = std::f32::consts::PI;
        let r: Rotor = Rotor::new(pi * 0.5, 0.3, -3., 1.);
        let t: Translator = Translator::translator(12., -2., 0.4, 1.);
        let m1: Motor = r * t;
        let l: Line = log(m1);
        let m2: Motor = exp(l);

        approx_eq(m1.scalar(), m2.scalar());
        approx_eq(m1.e12(), m2.e12());
        approx_eq(m1.e31(), m2.e31());
        approx_eq(m1.e23(), m2.e23());
        approx_eq(m1.e01(), m2.e01());
        approx_eq(m1.e02(), m2.e02());
        approx_eq(m1.e03(), m2.e03());
        approx_eq(m1.e0123(), m2.e0123());


        let m3: Motor = sqrt(m1) * sqrt(m1);
        approx_eq(m1.scalar(), m3.scalar());
        approx_eq(m1.e12(), m3.e12());
        approx_eq(m1.e31(), m3.e31());
        approx_eq(m1.e23(), m3.e23());
        approx_eq(m1.e01(), m3.e01());
        approx_eq(m1.e02(), m3.e02());
        approx_eq(m1.e03(), m3.e03());
        approx_eq(m1.e0123(), m3.e0123());
    }

    #[test]
    fn motor_slerp() {
        // Construct a motor from a translator and rotor
        let pi = std::f32::consts::PI;
        let r: Rotor = Rotor::new(pi * 0.5, 0.3, -3., 1.);
        let t: Translator = Translator::translator(12., -2., 0.4, 1.);
        let m1: Motor = r * t;
        let l: Line = log(m1);
        // Divide the motor action into three equal steps
        let step: Line = l / 3;
        let m_step: Motor = exp(step);
        let m2: Motor = m_step * m_step * m_step;
        approx_eq(m1.scalar(), m2.scalar());
        approx_eq(m1.e12(), m2.e12());
        approx_eq(m1.e31(), m2.e31());
        approx_eq(m1.e23(), m2.e23());
        approx_eq(m1.e01(), m2.e01());
        approx_eq(m1.e02(), m2.e02());
        approx_eq(m1.e03(), m2.e03());
        approx_eq(m1.e0123(), m2.e0123());
    }

    #[test]
    fn motor_blend() {
        let pi = std::f32::consts::PI;
        let r1: Rotor = Rotor::new(pi * 0.5, 0., 0., 1.);
        let t1: Translator = Translator::translator(1., 0., 0., 1.);
        let m1: Motor = r1 * t1;

        let r2 = Rotor::new(pi * 0.5, 0.3, -3., 1.);
        let t2 = Translator::translator(12., -2., 0.4, 1.);
        let m2: Motor = r2 * t2;

        let motion: Motor = m2 * m1.reverse();
        let step: Line = log(motion) / 4.;
        let motor_step: Motor = exp(step);

        // Applying motor_step 0 times to m1 is m1.
        // Applying motor_step 4 times to m1 is m2 * ~m1;
        let result: Motor = motor_step * motor_step * motor_step * motor_step * m1;
        approx_eq(result.scalar(), m2.scalar());
        approx_eq(result.e12(), m2.e12());
        approx_eq(result.e31(), m2.e31());
        approx_eq(result.e23(), m2.e23());
        approx_eq(result.e01(), m2.e01());
        approx_eq(result.e02(), m2.e02());
        approx_eq(result.e03(), m2.e03());
        approx_eq(result.e0123(), m2.e0123());
    }
}
