mod detail;
mod dual;
mod exp_log_sqrt;
mod geometric_product;
mod inner_product;
mod join;
mod line;
mod mat4x4;
mod meet;
mod motor;
mod plane;
mod point;
mod projection;
mod rotor;
mod translator;
mod util;

pub use dual::Dual;
pub use line::{Branch, IdealLine, Line};
pub use motor::Motor;
pub use plane::Plane;
pub use point::{Element, Point};
pub use rotor::{EulerAngles, Rotor};
pub use translator::Translator;
pub use util::{ApplyOp, ApplyToMany};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
