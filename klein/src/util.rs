pub trait ApplyOp<O> {
    fn apply_to(self, other: O) -> O;
}
