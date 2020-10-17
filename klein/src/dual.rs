use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Dual Numbers
/// A dual number is a multivector of the form $p + q\mathbf{e}_{0123}$.

#[derive(Copy, Clone, Default, Debug)]
pub struct Dual {
    pub p: f32,
    pub q: f32,
}

impl Dual {
    pub fn scalar(self) -> f32 {
        self.p
    }
    pub fn e0123(self) -> f32 {
        self.q
    }
}

impl AddAssign for Dual {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.p += rhs.p;
        self.q += rhs.q;
    }
}

impl SubAssign for Dual {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.p -= rhs.p;
        self.q -= rhs.q;
    }
}

impl MulAssign<f32> for Dual {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        self.p *= s;
        self.q *= s;
    }
}

impl DivAssign<f32> for Dual {
    #[inline]
    fn div_assign(&mut self, s: f32) {
        self.p /= s;
        self.q /= s;
    }
}

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Dual {
            p: self.p + rhs.p,
            q: self.q + rhs.q,
        }
    }
}

/// Ideal line subtraction
impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Dual {
            p: self.p - rhs.p,
            q: self.q - rhs.q,
        }
    }
}

impl Mul<f32> for Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Dual {
            p: self.p * s,
            q: self.q * s,
        }
    }
}
impl Mul<Dual> for f32 {
    type Output = Dual;
    #[inline]
    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            p: rhs.p * self,
            q: rhs.q * self,
        }
    }
}

impl Div<f32> for Dual {
    type Output = Dual;
    #[inline]
    fn div(self, s: f32) -> Self {
        Dual {
            p: self.p / s,
            q: self.q / s,
        }
    }
}
