use crate::CoeffType;

#[derive(Clone, Debug, PartialEq)]
pub struct Polynomial {
    pub(crate) coeffs: Vec<CoeffType>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<CoeffType>) -> Self {
        Self { coeffs }
    }

    pub fn empty() -> Self {
        Self { coeffs: vec![] }
    }

    /// Adds two polynomials in the ring.
    pub fn add(&self, poly: &Polynomial, coeff_modulus: Option<u128>) -> Polynomial {
        let mut result = vec![0; self.coeffs.len()];
        for i in 0..self.coeffs.len() {
            result[i] = self.coeffs[i] + poly.coeffs[i];
            if let Some(modulus) = coeff_modulus {
                result[i] %= modulus as CoeffType;
            }
        }
        Polynomial::new(result)
    }

    /// Subtracts second polynomial from first polynomial in the ring.
    pub fn subtract(&self, poly: &Polynomial, coeff_modulus: Option<u128>) -> Polynomial {
        let mut result = vec![0; self.coeffs.len()];
        for i in 0..self.coeffs.len() {
            result[i] = self.coeffs[i] - poly.coeffs[i];
            if let Some(modulus) = coeff_modulus {
                // TODO: This is most likely incorrect
                result[i] = (result[i] % modulus as CoeffType + modulus as CoeffType)
                    % modulus as CoeffType;
                // Ensure positive result
            }
        }
        Polynomial::new(result)
    }

    /// Multiplies a polynomial by another polynomial.
    pub fn multiply(
        &self,
        poly: &Polynomial,
        scaling_factor: usize,
        coeff_modulus: Option<u128>,
    ) -> Polynomial {
        assert!(self.coeffs.len() == poly.coeffs.len());
        let mut result = vec![0; self.coeffs.len()];
        for i in 0..self.coeffs.len() {
            result[i] = self.coeffs[i] * poly.coeffs[i];
            result[i] >>= scaling_factor;
            if let Some(modulus) = coeff_modulus {
                result[i] %= modulus as CoeffType;
            }
        }
        Polynomial::new(result)
    }

    /// Multiplies polynomial by a scalar.
    pub fn scalar_multiply(&self, scalar: CoeffType, coeff_modulus: Option<u128>) -> Polynomial {
        let mut new_coeffs = vec![0; self.coeffs.len()];
        for i in 0..self.coeffs.len() {
            new_coeffs[i] = self.coeffs[i] * scalar;
            if let Some(modulus) = coeff_modulus {
                new_coeffs[i] %= modulus as CoeffType;
            }
        }
        Polynomial::new(new_coeffs)
    }

    /// Divides polynomial by a scalar.
    pub fn scalar_integer_divide(
        &self,
        scalar: CoeffType,
        coeff_modulus: Option<CoeffType>,
    ) -> Polynomial {
        let mut new_coeffs = vec![0; self.coeffs.len()];
        for i in 0..self.coeffs.len() {
            new_coeffs[i] = self.coeffs[i] / scalar;
            if let Some(modulus) = coeff_modulus {
                new_coeffs[i] = (new_coeffs[i] % modulus + modulus) % modulus; // Ensure positive result
            }
        }
        Polynomial::new(new_coeffs)
    }

    /// Mods all coefficients in the given coefficient modulus.
    pub fn modulo(&self, coeff_modulus: i64) -> Polynomial {
        let new_coeffs: Vec<i64> = self
            .coeffs
            .iter()
            .map(|&c| (c % coeff_modulus + coeff_modulus) % coeff_modulus)
            .collect();
        Polynomial::new(new_coeffs)
    }

    /// Evaluates the polynomial at the given input value.
    pub fn evaluate(&self, inp: i64) -> i64 {
        let mut result = self.coeffs[self.coeffs.len() - 1];
        for i in (0..self.coeffs.len() - 1).rev() {
            result = result * inp + self.coeffs[i];
        }
        result
    }

    pub fn concatenate(&self, poly: &Polynomial) -> Polynomial {
        let mut result = Vec::new();
        result.extend_from_slice(&self.coeffs);
        result.extend_from_slice(&poly.coeffs);
        Polynomial::new(result)
    }
}
