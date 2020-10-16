#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 4x4 column-major matrix (used for converting rotors/motors to matrix form to
/// upload to shaders).
enum Mat4x4 {
	Cols([__m128;  4]),
	Data([f32   ; 16])
}
