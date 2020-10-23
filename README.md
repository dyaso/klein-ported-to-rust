# klein-ported-to-rust
Untried (as of mid October 2020, though all the tests pass) Rust port of Jeremy Ong's [SIMD-enabled plane-based geometric algebra library, "Klein"](https://www.jeremyong.com/klein/). Most intrinsics code came via [the C# version](https://github.com/Ziriax/KleinSharp) which had already gone through the C++ templates.

Run `cargo run --example druid_3d_shapes` in the `demo` directory to see an example drawn to a [Druid](https://github.com/linebender/druid) window.

## Porting notes

I'm uncertain about Rust conventions so please open Issues or contact me on [bivector.net discord](https://discord.gg/vGY6pPk) if something's unusual. This is a Rust learning project (as well as SIMD use and PGA itself)!

C++'s multiple constructors per class are replaced by `default()`, `new( ... )` for an appropriate convenience constructor taking `f32`s as arguments, `from( ... )` where constructing from a single object makes sense, and e.g. `Motor::from_rotor_and_translator( ... )`, `Motor::from_screw_line(angle_rad: f32, dist: f32, l: Line)` when construction from several objects at once makes sense.

Rust doesn't let you overload the function call operator, so if you want to e.g. apply a rotation please say `let rotated = rotation.`**`apply_to`**`(something);` instead.

The `Point` struct has extra methods `scale` and `scaled` (former mutates in place, latter returns new Point) which perform coordinate scaling.

The `Mat4x4` struct has a `perspective` projection matrix constructor method.


