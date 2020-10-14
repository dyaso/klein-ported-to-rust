use super::detail::sse::{hi_dp, hi_dp_bc, rsqrt_nr1, sqrt_nr1};
use super::detail::sandwich::{sw00, sw30};
use super::point::Point;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// In projective geometry, planes are the fundamental element through which all
/// other entities are constructed. Lines are the meet of two planes, and points
/// are the meet of three planes (equivalently, a line and a plane).
///
/// The plane multivector in PGA looks like
/// $d\mathbf{e}_0 + a\mathbf{e}_1 + b\mathbf{e}_2 + c\mathbf{e}_3$. Points
/// that reside on the plane satisfy the familiar equation
/// $d + ax + by + cz = 0$.
#[derive(Copy, Clone)]
pub struct Plane {
    p0_: __m128,
}

impl Plane {
    /// The constructor performs the rearrangement so the plane can be specified
    /// in the familiar form: ax + by + cz + d
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Plane {
        Plane {
            p0_: unsafe { _mm_set_ps(c, b, a, d) },
        }
    }
}

impl From<__m128> for Plane {
    fn from(xmm: __m128) -> Plane {
        Plane { p0_: xmm }
    }
}

/// Data should point to four floats with memory layout `(d, a, b, c)` where
/// `d` occupies the lowest address in memory.
impl From<&f32> for Plane {
    fn from(data: &f32) -> Plane {
        Plane {
            p0_: unsafe { _mm_loadu_ps(data) },
        }
    }
}

/// Unaligned load of data. The `data` argument should point to 4 floats
/// corresponding to the
/// `(d, a, b, c)` components of the plane multivector where `d` occupies
/// the lowest address in memory.
///
/// !!! tip
///
/// This is a faster mechanism for setting data compared to setting
///     components one at a time.
pub fn load(data: &f32) -> Plane {
    Plane {
        p0_: unsafe { _mm_loadu_ps(data) },
    }
}

impl Plane {
    /// Normalize this plane $p$ such that $p \cdot p = 1$.
    ///
    /// In order to compute the cosine of the angle between planes via the
    /// inner product operator `|`, the planes must be normalized. Producing a
    /// normalized rotor between two planes with the geometric product `*` also
    /// requires that the planes are normalized.
    pub fn normalize(mut self) {
        unsafe {
            let mut inv_norm = rsqrt_nr1(hi_dp_bc(self.p0_, self.p0_));
            #[cfg(target_feature = "sse4.1")]
            {
                inv_norm = _mm_blend_ps(inv_norm, _mm_set_ss(1.0), 1);
            }
            #[cfg(not(target_feature = "sse4.1"))]
            {
                inv_norm = _mm_add_ps(inv_norm, _mm_set_ss(1.0));
            }
            self.p0_ = _mm_mul_ps(inv_norm, self.p0_);
        }
    }

    /// Return a normalized copy of this plane.
    pub fn normalized(self) -> Plane {
        // ???
        //    	let out = Plane::from(self);
        let out = self;
        out.normalize();
        return out;
    }

    /// Compute the plane norm, which is often used to compute distances
    /// between points and lines.
    ///
    /// Given a normalized point $P$ and normalized line $\ell$, the plane
    /// $P\vee\ell$ containing both $\ell$ and $P$ will have a norm equivalent
    /// to the distance between $P$ and $\ell$.
    pub fn norm(self) -> f32 {
        let mut out: f32 = 0.0;
        unsafe {
            _mm_store_ss(&mut out, sqrt_nr1(hi_dp(self.p0_, self.p0_)));
        }
        return out;
    }

    pub fn invert(mut self) {
        unsafe {
            let inv_norm = rsqrt_nr1(hi_dp_bc(self.p0_, self.p0_));
            self.p0_ = _mm_mul_ps(inv_norm, self.p0_);
            self.p0_ = _mm_mul_ps(inv_norm, self.p0_);
        }
    }

    pub fn inverse(self) -> Plane {
        let out = self;
        out.invert();
        return out;
    }
}

impl PartialEq for Plane {
    fn eq(&self, other: &Plane) -> bool {
        unsafe {return _mm_movemask_ps(_mm_cmpeq_ps(self.p0_, other.p0_)) == 0b1111;}
    }
}

impl Plane {
    pub fn approx_eq(self, other: &Plane, epsilon: f32) -> bool
    {
        unsafe{
        let eps = _mm_set1_ps(epsilon);
        let cmp = _mm_cmplt_ps(
            _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(self.p0_, other.p0_)), eps);
        return _mm_movemask_ps(cmp) == 0b1111;
    }
    }
}

trait ApplyPlaneOp<O> {
    fn apply_to(self, other: O) -> O;
}

impl ApplyPlaneOp<Plane> for Plane {
    /// Reflect another plane $p_2$ through this plane $p_1$. The operation
    /// performed via this call operator is an optimized routine equivalent to
    /// the expression $p_1 p_2 p_1$.
    ///
    /// rust port comment - cannot overload the function call operator in rust
    /// [[nodiscard]] plane KLN_VEC_CALL operator()(plane const& p) const noexcept
    fn apply_to (self, p: Plane) -> Plane
    {
        let mut out = self;
        sw00(self.p0_, p.p0_, &mut out.p0_);
        return out;
    }
}

impl ApplyPlaneOp<Point> for Plane {
    /// Reflect the point $P$ through this plane $p$. The operation
    /// performed via this call operator is an optimized routine equivalent to
    /// the expression $p P p$.
    fn apply_to (self, p: Point) -> Point
    {
        return Point::from(sw30(self.p0_, p.p3_));
    }

}



