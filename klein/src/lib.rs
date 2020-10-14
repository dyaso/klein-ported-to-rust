mod detail;
mod plane;
mod point;
mod line;
mod meet;
mod join;
mod dual;

pub use plane::Plane;
pub use point::Point;
pub use line::{Line,Branch,IdealLine};
pub use dual::Dual;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
