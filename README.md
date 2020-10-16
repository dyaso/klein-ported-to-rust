# klein-ported-to-rust
Incomplete (as of mid October 2020) Rust port of Jeremy Ong's [SIMD-enabled projective geometric algebra library, "Klein"](https://www.jeremyong.com/klein/).

Run `cargo run --example z_plane` to see a 2D example drawn using the Druid UI toolkit to obtain a window.

## Porting notes

Rust doesn't seem to let you overload the function call operator syntax, so if you want to e.g. apply a rotation to something please use `let rotated = rotation.`**`apply_to`**`(something);` instead.

I also can't seem to overload functions based on argument type, so `exp`, `log`, and `sqrt` are currently methods of the value you want to use as an operand.

