pub trait ApplyOp<O> {
    fn apply_to(self, other: O) -> O;
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

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
