#[macro_use]
mod util;

mod detail;
mod dual;
mod exp_log_sqrt;
mod geometric_product;
mod inner_product;
mod join;
mod line;
mod branch;
mod ideal_line;
mod matrices;
mod meet;
mod motor;
mod plane;
mod point;
mod projection;
mod rotor;
mod translator;
mod direction;

pub use dual::Dual;
pub use line::{Line};
pub use ideal_line::{IdealLine};
pub use branch::{Branch};
pub use matrices::{Mat3x4, Mat4x4};
pub use motor::Motor;
pub use plane::Plane;
pub use point::{Element, Point, origin};
pub use direction::{Direction};
pub use rotor::{EulerAngles, Rotor};
pub use translator::Translator;
pub use util::{ApplyTo, ApplyToMany};
pub use exp_log_sqrt::{log, sqrt, exp};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
