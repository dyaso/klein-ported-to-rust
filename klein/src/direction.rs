#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::detail::sse::{dp_bc, hi_dp, hi_dp_bc, rcp_nr1, rsqrt_nr1};

#[derive(Copy, Clone, Debug)]
pub struct Direction {
    pub p3_: __m128,
}

use std::ops::Neg;

impl Direction {
    pub fn new(x: f32, y: f32, z: f32) -> Direction {
        unsafe {
            let mut out = Direction{
                p3_: _mm_set_ps(z, y, x, 0.),
            };
            out.normalize();
            out
        }
    }

    get_basis_blade_fn!(e032, e023, p3_, 1);
    get_basis_blade_fn!(e013, e031, p3_, 2);
    get_basis_blade_fn!(e021, e012, p3_, 3);

    /// Normalize this direction by dividing all components by the
    /// magnitude (by default, `rsqrtps` is used with a single Newton-Raphson
    /// refinement iteration)
    pub fn normalize(&mut self)    {
    	unsafe {
	        let tmp = rsqrt_nr1(hi_dp_bc(self.p3_, self.p3_));
    	    self.p3_        = _mm_mul_ps(self.p3_, tmp);
    	}
    }


}