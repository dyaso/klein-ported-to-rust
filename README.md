# klein-ported-to-rust
Incomplete (as of mid October 2020) Rust port of Jeremy Ong's [SIMD-enabled projective geometric algebra library, "Klein"](https://www.jeremyong.com/klein/).

Run `cargo run --example z_plane` to see a 2D example.

## Porting notes

Rust doesn't seem to let you overload the function call operator syntax, so if you want to e.g. apply a rotation to something use `let rotated = rotation.`**`apply_to`**`(something);` instead.



