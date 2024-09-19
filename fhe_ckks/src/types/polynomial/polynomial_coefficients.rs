use log::debug;
use num_traits::{PrimInt, Signed};
use std::ops::{Add, Mul, Rem, Shr, Sub};

use super::double_size::DoubleSized;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolynomialCoefficients<T>
where
    T: PrimInt + Signed + Clone + Copy,
{
    pub(super) coeffs: T,
    pub(super) has_data: bool,
}

impl<T> PolynomialCoefficients<T>
where
    T: PrimInt + Signed + Clone + Copy,
{
    pub fn new(coeffs: T) -> Self {
        Self {
            coeffs,
            has_data: true,
        }
    }

    pub fn default() -> Self {
        Self {
            coeffs: T::zero(),
            has_data: false,
        }
    }

    pub fn value(&self) -> T {
        self.coeffs
    }

    // pub fn store(&mut self, coeffs: T) {
    //     self.coeffs = coeffs;
    //     self.has_data = true;
    // }

    // pub fn clear(&mut self) {
    //     self.coeffs = T::zero();
    //     self.has_data = false;
    // }
}

impl<T> Add for PolynomialCoefficients<T>
where
    T: PrimInt + Signed + Clone + Copy,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        #[cfg(feature = "debug")]
        if !self.has_data || !other.has_data {
            debug!("Adding uninitialized coefficients");
        }

        Self {
            coeffs: self.coeffs + other.coeffs,
            has_data: true,
        }
    }
}

impl<T> Add<T> for PolynomialCoefficients<T>
where
    T: PrimInt + Signed + Clone + Copy,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        #[cfg(feature = "debug")]
        if !self.has_data {
            debug!("Adding uninitialized coefficients");
        }

        Self {
            coeffs: self.coeffs + rhs,
            has_data: true,
        }
    }
}

impl<T> Sub for PolynomialCoefficients<T>
where
    T: PrimInt + Signed + Clone + Copy,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        #[cfg(feature = "debug")]
        if !self.has_data || !other.has_data {
            debug!("Subtracting uninitialized coefficients");
        }

        Self {
            coeffs: self.coeffs - other.coeffs,
            has_data: true,
        }
    }
}

impl<T> Mul for PolynomialCoefficients<T>
where
    T: PrimInt + Signed + Clone + Copy,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        #[cfg(feature = "debug")]
        if !self.has_data || !other.has_data {
            debug!("Multiplying uninitialized coefficients");
        }

        Self {
            coeffs: self.coeffs * other.coeffs,
            has_data: true,
        }
    }
}

impl<T> Rem for PolynomialCoefficients<T>
where
    T: PrimInt + Signed + Clone + Copy,
{
    type Output = Self;

    fn rem(self, rem: PolynomialCoefficients<T>) -> Self {
        #[cfg(feature = "debug")]
        if !self.has_data {
            debug!("Remainder of uninitialized coefficients");
        }

        Self {
            coeffs: self.coeffs % rem.coeffs,
            has_data: true,
        }
    }
}

impl<T> Rem<T> for PolynomialCoefficients<T>
where
    T: PrimInt + Signed + Clone + Copy,
{
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        #[cfg(feature = "debug")]
        if !self.has_data {
            debug!("Remainder of uninitialized coefficients");
        }

        Self {
            coeffs: self.coeffs % rhs,
            has_data: true,
        }
    }
}

impl<T> Shr<usize> for PolynomialCoefficients<T>
where
    T: PrimInt + Signed + Clone + Copy,
{
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        #[cfg(feature = "debug")]
        if !self.has_data {
            debug!("Shifting uninitialized coefficients");
        }

        Self {
            coeffs: self.coeffs >> rhs,
            has_data: true,
        }
    }
}

impl<T> PolynomialCoefficients<T>
where
    T: DoubleSized + PrimInt + Signed + Clone + Copy,
{
    pub fn to_double(&self) -> PolynomialCoefficients<T::Double> {
        PolynomialCoefficients {
            coeffs: T::to_double(self.coeffs),
            has_data: self.has_data,
        }
    }

    pub fn from_double(double: PolynomialCoefficients<T::Double>) -> Self {
        PolynomialCoefficients {
            coeffs: T::from_double(double.coeffs),
            has_data: double.has_data,
        }
    }
}

// impl<T> PolynomialCoefficients<T::Double>
// where
//     T: DoubleSized + PrimInt + Signed + Debug,
//     T::Double: PrimInt + Signed + Debug,
// {
//     pub fn from_double(self) -> PolynomialCoefficients<T> {
//         PolynomialCoefficients {
//             coeffs: T::from_double(self.coeffs),
//             has_data: self.has_data,
//         }
//     }
// }
