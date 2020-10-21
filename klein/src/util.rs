
pub trait ApplyTo<O> {
    fn apply_to(self, other: O) -> O;
}

pub trait ApplyToMany<O> {
    fn apply_to_many(self, input: &[O], other: &mut [O], count: usize);
}

#[macro_use]
macro_rules! get_basis_blade_fn {
    ($name:ident, $reverse_name:ident, $component:ident, $index:expr) => {
        pub fn $name(self) -> f32 {
            let mut out = <[f32; 4]>::default();
            unsafe {
                _mm_store_ps(&mut out[0], self.$component);
            }
            out[$index]
        }

        pub fn $reverse_name(self) -> f32 {
            -self.$name()
        }
    };
}

macro_rules! common_operations {

    ($object:ty, $component:ident) 
    => {
        impl From<__m128> for $object {
            fn from(xmm: __m128) -> Self {
                Self { $component: xmm }
            }
        }

        impl From<$object> for __m128 {
            fn from(me :$object) -> Self {
                me.$component
            }
        }


        impl Default for $object {
            fn default() -> Self {
                unsafe {
                    Self {$component: _mm_setzero_ps()}
                }
            }
        }
        use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign, Add, Sub, Div, Mul};
    
        impl<T: Into<f32>> MulAssign<T> for $object {
            #[inline]
            fn mul_assign(&mut self, s: T) {
                unsafe {
                    self.$component = _mm_mul_ps(self.$component, _mm_set1_ps(s.into()));
                }
            }
        }


        impl AddAssign for $object {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                unsafe {
                    self.$component = _mm_add_ps(self.$component, rhs.$component);
                }
            }
        }
        
        impl SubAssign for $object {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                unsafe { self.$component = _mm_sub_ps(self.$component, rhs.$component) }
            }
        }
        
        impl Add for $object {
            type Output = $object;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                unsafe { Self::from(_mm_add_ps(self.$component, rhs.$component)) }
            }
        }

        impl Sub for $object {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                unsafe { Self::from(_mm_sub_ps(self.$component, rhs.$component)) }
            }
        }


        impl<T: Into<f32>> Mul<T> for $object {
            type Output = Self;
            #[inline]
            fn mul(self, s: T) -> Self {
                unsafe { Self::from(_mm_mul_ps(self.$component, _mm_set1_ps(s.into()))) }
            }
        }

        impl<T: Into<f32>> DivAssign<T> for $object {
            #[inline]
            fn div_assign(&mut self, s: T) {
                unsafe {
                    self.$component = _mm_mul_ps(self.$component, rcp_nr1(_mm_set1_ps(s.into())));
                }
            }
        }
        


        impl $object {
            #[inline]
            pub fn normalized(self) -> Self        {
                let mut out = Self::from(self.$component);
                out.normalize();
                out
            }
    
            #[inline]
            pub fn reverse(self) -> Self {
                unsafe {
                    let flip: __m128 = _mm_set_ps(-0., -0., -0., 0.);
                    Self::from(_mm_xor_ps(self.$component, flip))
                }
            }
        }

    }
}

#[cfg(test)]
mod tests {
    #![cfg(target_arch = "x86_64")]

    #[allow(dead_code)]
    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6)
    }

    // fn test_log2()	{
    //     let x = 12;
    //     assert_eq!(uint::log2(8), 3);
    //     assert_eq!(uint::log2(9), 3);
    //     assert_eq!(uint::log2(x), 3);
    //     // uint64_t y = 1ull << 34;
    //     // CHECK_EQ(kln::log2(y), 34);
    //     // CHECK_EQ(kln::log2((1ull << 34) | (1 << 12)), 34);
    // }
}
