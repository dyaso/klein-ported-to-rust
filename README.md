# klein-ported-to-rust
Untried (as of mid October 2020; all the tests pass) Rust port of Jeremy Ong's [SIMD-enabled plane-based projective geometric algebra library, "Klein"](https://www.jeremyong.com/klein/). Most intrinsics code came via [the C# version](https://github.com/Ziriax/KleinSharp) which had already gone through the C++ templates and expanded the preprocessor macros.

Run `cargo run --example spinning_tetrahedron` in the `demo` directory to see an example drawn to a [Druid](https://github.com/linebender/druid) window.

## Porting notes

I'm uncertain about typical Rust conventions so please open Issues or contact me on [bivector.net discord](https://discord.gg/vGY6pPk) if something's unusual. This is a Rust learning project (as well as SIMD use and PGA itself).

Rust doesn't let you overload the function call operator, so if you want to e.g. apply a rotation please `use` the `ApplyTo` trait and say `let rotated = rotation.`**`apply_to`**`(something);` instead.

