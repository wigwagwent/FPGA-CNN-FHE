use double_size::DoubleSized;
use num_traits::{PrimInt, Signed};
use polynomial_coefficients::PolynomialCoefficients;

pub mod double_size;
mod polynomial_coefficients;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Polynomial<T, const N: usize>
where
    T: PrimInt + Signed + Clone + Copy,
{
    pub(crate) coeffs: [PolynomialCoefficients<T>; N],
}

impl<T, const N: usize> Polynomial<T, N>
where
    T: PrimInt + Signed + Clone + Copy + DoubleSized,
{
    pub fn new(coeffs: Vec<T>) -> Self {
        assert!(
            coeffs.len() <= N,
            "{}",
            format!("Expected {} coefficients or less", N)
        );
        let mut array = [PolynomialCoefficients::default(); N];
        for (i, coeff) in coeffs.into_iter().enumerate() {
            array[i] = PolynomialCoefficients::new(coeff);
        }
        Self { coeffs: array }
    }

    pub fn default() -> Self {
        Self {
            coeffs: [PolynomialCoefficients::default(); N],
        }
    }

    pub fn len(&self) -> usize {
        N
    }

    pub fn used_len(&self) -> usize {
        self.coeffs.iter().filter(|c| c.has_data).count()
    }

    /// Adds two polynomials in the ring.
    pub fn add(
        &self,
        poly: &Polynomial<T, N>,
        coeff_modulus: Option<T::Double>,
    ) -> Polynomial<T, N> {
        let mut result = [PolynomialCoefficients::default(); N];

        for i in 0..N {
            let sum = self.coeffs[i].to_double() + poly.coeffs[i].to_double();
            let modded_sum = if let Some(modulus) = coeff_modulus {
                let mod_sum = sum % modulus;
                (mod_sum + modulus) % modulus
            } else {
                sum
            };
            result[i] = PolynomialCoefficients::<T>::from_double(modded_sum); //TODO: Make this modded_sum.from_double()
        }

        Polynomial { coeffs: result }
    }

    /// Subtracts two polynomials in the ring.
    pub fn subtract(
        &self,
        poly: &Polynomial<T, N>,
        coeff_modulus: Option<T::Double>,
    ) -> Polynomial<T, N> {
        let mut result = [PolynomialCoefficients::default(); N];

        for i in 0..N {
            let diff = self.coeffs[i].to_double() - poly.coeffs[i].to_double();

            let modded_diff = if let Some(modulus) = coeff_modulus.clone() {
                let mod_diff = diff % modulus;
                (mod_diff + modulus) % modulus
            } else {
                diff
            };

            result[i] = PolynomialCoefficients::<T>::from_double(modded_diff);
        }

        Polynomial { coeffs: result }
    }

    /// Multiplies two polynomials in the ring.
    pub fn multiply(
        &self,
        poly: &Polynomial<T, N>,
        scaling_factor: usize,
        coeff_modulus: Option<T::Double>,
    ) -> Polynomial<T, N> {
        let mut result = [PolynomialCoefficients::default(); N];

        for i in 0..N {
            let product = self.coeffs[i].to_double() * poly.coeffs[i].to_double();
            let scaled_product = product >> scaling_factor;

            let modded_product = if let Some(modulus) = coeff_modulus.clone() {
                let mod_product = scaled_product % modulus;
                (mod_product + modulus) % modulus
            } else {
                scaled_product
            };

            result[i] = PolynomialCoefficients::<T>::from_double(modded_product);
        }

        Polynomial { coeffs: result }
    }

    /// Rotates the polynomial to the right by the given amount.
    pub fn rotate_right(&self, amount: usize) -> Polynomial<T, N> {
        let mut result = [PolynomialCoefficients::default(); N];

        for i in 0..N {
            result[i] = self.coeffs[(i + amount) % N];
        }

        Polynomial { coeffs: result }
    }

    /// Turns all coefficients in the given coefficient modulus
    /// to the range (-q/2, q/2].
    /// Turns all coefficients of the current polynomial
    /// in the given coefficient modulus to the range (-q/2, q/2].
    /// Args:
    ///     coeff_modulus (int): Modulus a of coefficients of polynomial
    ///         ring R_a.
    /// Returns:
    ///     A Polynomial whose coefficients are modulo coeff_modulus.
    pub fn mod_small(&self, coeff_modulus: Option<T::Double>) -> Polynomial<T, N> {
        let mut new_coeffs = [PolynomialCoefficients::default(); N];
        for i in 0..N {
            let coeff = self.coeffs[i].to_double();
            let modded_coeff = if let Some(modulus) = coeff_modulus.clone() {
                let mod_coeff = coeff % modulus;
                (mod_coeff + modulus) % modulus
            } else {
                coeff
            };
            new_coeffs[i] = PolynomialCoefficients::<T>::from_double(modded_coeff);
        }

        Polynomial { coeffs: new_coeffs }
    }

    /// Multiplies polynomial by a scalar.
    pub fn scalar_multiply(&self, scalar: T, coeff_modulus: Option<T::Double>) -> Polynomial<T, N> {
        let mut new_coeffs = [PolynomialCoefficients::default(); N];
        for i in 0..N {
            let product = self.coeffs[i].to_double() * scalar.to_double();
            let modded_product = if let Some(modulus) = coeff_modulus.clone() {
                let mod_product = product % modulus;
                (mod_product + modulus) % modulus
            } else {
                product
            };
            new_coeffs[i] = PolynomialCoefficients::<T>::from_double(modded_product);
        }

        Polynomial { coeffs: new_coeffs }
    }
    // pub fn scalar_multiply(&self, scalar: CoeffType, coeff_modulus: Option<ModType>) -> Polynomial {
    //     let mut new_coeffs = vec![0; self.coeffs.len()];
    //     for i in 0..self.coeffs.len() {
    //         new_coeffs[i] = self.coeffs[i] * scalar;
    //         if let Some(modulus) = coeff_modulus {
    //             new_coeffs[i] %= modulus as CoeffType;
    //         }
    //     }
    //     Polynomial::new(new_coeffs)
    // }

    // /// Divides polynomial by a scalar.
    // pub fn scalar_integer_divide(
    //     &self,
    //     scalar: CoeffType,
    //     coeff_modulus: Option<CoeffType>,
    // ) -> Polynomial {
    //     let mut new_coeffs = vec![0; self.coeffs.len()];
    //     for i in 0..self.coeffs.len() {
    //         new_coeffs[i] = self.coeffs[i] / scalar;
    //         if let Some(modulus) = coeff_modulus {
    //             new_coeffs[i] = (new_coeffs[i] % modulus + modulus) % modulus; // Ensure positive result
    //         }
    //     }
    //     Polynomial::new(new_coeffs)
    // }

    // /// Mods all coefficients in the given coefficient modulus.
    // pub fn modulo(&self, coeff_modulus: i64) -> Polynomial {
    //     let new_coeffs: Vec<i64> = self
    //         .coeffs
    //         .iter()
    //         .map(|&c| (c % coeff_modulus + coeff_modulus) % coeff_modulus)
    //         .collect();
    //     Polynomial::new(new_coeffs)
    // }

    // /// Evaluates the polynomial at the given input value.
    // pub fn evaluate(&self, inp: i64) -> i64 {
    //     let mut result = self.coeffs[self.coeffs.len() - 1];
    //     for i in (0..self.coeffs.len() - 1).rev() {
    //         result = result * inp + self.coeffs[i];
    //     }
    //     result
    // }
}
