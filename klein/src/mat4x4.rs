#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 4x4 column-major matrix (used for converting rotors/motors to matrix form to
/// upload to shaders).
enum mat4x4 {
	cols([__m128;  4]),
	data([f32   ; 16])
}
