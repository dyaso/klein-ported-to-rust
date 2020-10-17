use crate::{Line, Plane, Point};

/// Projections
///
/// Projections in Geometric Algebra take on a particularly simple form.
/// For two geometric entities $a$ and $b$, there are two cases to consider.
/// First, if the grade of $a$ is greater than the grade of $b$, the projection
/// of $a$ on $b$ is given by:
///
/// $$ \textit{proj}_b a = (a \cdot b) \wedge b $$
///
/// The inner product can be thought of as the part of $b$ _least like_ $a$.
/// Using the meet operator on this part produces the part of $b$ _most like_
/// $a$. A simple sanity check is to consider the grades of the result. If the
/// grade of $b$ is less than the grade of $a$, we end up with an entity with
/// grade $a - b + b = a$ as expected.
///
/// In the second case (the grade of $a$ is less than the grade of $b$), the
/// projection of $a$ on $b$ is given by:
///
/// $$ \textit{proj}_b a = (a \cdot b) \cdot b $$
///
/// It can be verified that as in the first case, the grade of the result is the
/// same as the grade of $a$. As this projection occurs in the opposite sense
/// from what one may have seen before, additional clarification is provided
/// below.

trait Project<O> {
    #[inline]
    fn project(self, other: O) -> Self;
}

macro_rules! project_onto_lower_grade {
    ( $a:ty, $b:ty ) => {
        impl Project<$b> for $a {
            fn project(self, b: $b) -> $a {
                let a = self;
                (a | b) ^ b
            }
        }
    };
}

project_onto_lower_grade!(Point, Line);

/// Project a point onto a plane
project_onto_lower_grade!(Point, Plane);

/// Project a line onto a plane
project_onto_lower_grade!(Line, Plane);

/// Project a plane onto a point. Given a plane $p$ and point $P$, produces the
/// plane through $P$ that is parallel to $p$.
///
/// Intuitively, the point is represented dually in terms of a _pencil of
/// planes_ that converge on the point itself. When we compute $p | P$, this
/// selects the line perpendicular to $p$ through $P$. Subsequently, taking the
/// inner product with $P$ again selects the plane from the plane pencil of $P$
/// _least like_ that line.

macro_rules! project_onto_higher_grade {
    ( $a:ty, $b:ty ) => {
        impl Project<$b> for $a {
            fn project(self, b: $b) -> $a {
                let a = self;
                (a | b) | b
            }
        }
    };
}

project_onto_higher_grade!(Plane, Point);

/// Project a line onto a point. Given a line $\ell$ and point $P$, produces the
/// line through $P$ that is parallel to $\ell$.
project_onto_higher_grade!(Line, Point);

/// Project a plane onto a line. Given a plane $p$ and line $\ell$, produces the
/// plane through $\ell$ that is parallel to $p$ if $p \parallel \ell$.
///
/// If $p \nparallel \ell$, the result will be the plane $p'$ containing $\ell$
/// that maximizes $p \cdot p'$ (that is, $p'$ is as parallel to $p$ as
/// possible).
project_onto_higher_grade!(Plane, Line);
