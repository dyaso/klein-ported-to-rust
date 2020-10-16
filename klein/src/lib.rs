mod detail;
mod dual;
mod join;
mod line;
mod meet;
mod plane;
mod point;
mod inner_product;
mod mat4x4;
mod translator;
mod rotor;
mod motor;
mod geometric_product;
mod util;

pub use dual::Dual;
pub use line::{Branch, IdealLine, Line};
pub use translator::Translator;
pub use rotor::{Rotor, EulerAngles};
pub use motor::Motor;
pub use plane::{Plane};
pub use point::Point;
pub use util::ApplyOp;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
