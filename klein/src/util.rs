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
