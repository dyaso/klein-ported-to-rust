# klein-ported-to-rust
Incomplete (as of mid October 2020) Rust port of Jeremy Ong's [SIMD-enabled projective geometric algebra library, "Klein"](https://www.jeremyong.com/klein/). Most intrinsics code ported from [the C# port](https://github.com/Ziriax/KleinSharp) which had already gone through the C++ templates and expanded the preprocessor macros.

Run `cargo run --example spinning_tetrahedron` to see an example drawn using a [Druid](https://github.com/linebender/druid) window.

Yet to be ported are functions relating to matrices (mostly useful for interacting with existing graphics libraries, less necessary as Klein can apply GA operations in shaders) and maths operations on the Motor type (which can be copied and pasted from the Line type).

## Porting notes

Rust doesn't seem to let you overload the function call operator syntax, so if you want to e.g. apply a rotation to something please use `let rotated = rotation.`**`apply_to`**`(something);` instead.

I also can't seem to overload functions based on argument type, so `exp`, `log`, and `sqrt` are currently methods on the operand value.

