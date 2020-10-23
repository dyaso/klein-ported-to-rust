#[macro_use]
mod util;

mod branch;
mod detail;
mod direction;
mod dual;
mod exp_log_sqrt;
mod geometric_product;
mod ideal_line;
mod inner_product;
mod join;
mod line;
mod matrices;
mod meet;
mod motor;
mod plane;
mod point;
mod projection;
mod rotor;
mod translator;

pub use branch::Branch;
pub use direction::Direction;
pub use dual::Dual;
pub use exp_log_sqrt::{exp, log, sqrt};
pub use ideal_line::IdealLine;
pub use line::Line;
pub use matrices::{Mat3x4, Mat4x4};
pub use motor::Motor;
pub use plane::Plane;
pub use point::{origin, Element, Point};
pub use rotor::{EulerAngles, Rotor};
pub use translator::Translator;
pub use util::{ApplyTo, ApplyToMany};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
