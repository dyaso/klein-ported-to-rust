mod detail;
mod dual;
mod join;
mod line;
mod meet;
mod plane;
mod point;
mod inner_product;
mod mat4x4;

pub use dual::Dual;
pub use line::{Branch, IdealLine, Line};
pub use plane::{Plane};
pub use point::Point;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
