# klein-ported-to-rust
Untried (as of mid October 2020; all the tests pass) Rust port of Jeremy Ong's [SIMD-enabled plane-based projective geometric algebra library, "Klein"](https://www.jeremyong.com/klein/). Most intrinsics code comes via [the C# version](https://github.com/Ziriax/KleinSharp) which has already gone through the C++ templates and expanded the preprocessor macros.

Run `cargo run --example spinning_tetrahedron` in the `demo` directory to see an example drawn to a [Druid](https://github.com/linebender/druid) window.

## Porting notes

Rust doesn't seem to let you overload the function call operator syntax, so if you want to e.g. apply a rotation to something please use `let rotated = rotation.`**`apply_to`**`(something);` instead.

I also can't seem to overload functions based on argument type, so `exp`, `log`, and `sqrt` are currently methods on the operand value.

