use crate::types::ciphertext::Ciphertext;
use crate::types::polynomial::Polynomial;
use crate::CoeffType;

/// A plaintext polynomial. This is a polynomial with integer coefficients that is used to represent
/// the plaintext data. The coefficients of the polynomial are typically small integers. The plaintext
/// polynomial is encrypted to produce a ciphertext polynomial.
#[derive(Clone, Debug, PartialEq)]
pub struct Plaintext {
    /// The polynomial representing the plaintext data.
    pub(crate) poly: Polynomial,
    /// The scaling factor of the plaintext. 2^scaling_factor is the factor by which the plaintext
    /// was scaled.
    pub(crate) scaling_factor: usize,
}

impl Plaintext {
    pub fn new(poly: Polynomial, scaling_factor: usize) -> Self {
        Self {
            poly,
            scaling_factor,
        }
    }

    pub fn add(&self, plaintext: &Plaintext) -> Plaintext {
        assert!(self.scaling_factor == plaintext.scaling_factor);
        let new_poly = self.poly.add(&plaintext.poly, None);
        Plaintext::new(new_poly, self.scaling_factor)
    }

    pub fn subtract(&self, plaintext: &Plaintext) -> Plaintext {
        assert!(self.scaling_factor == plaintext.scaling_factor);
        let new_poly = self.poly.subtract(&plaintext.poly, None);
        Plaintext::new(new_poly, self.scaling_factor)
    }

    pub fn scalar_multiply(&self, scalar: CoeffType) -> Plaintext {
        let new_poly = self.poly.scalar_multiply(scalar, None);
        Plaintext::new(new_poly, self.scaling_factor)
    }

    pub fn scalar_integer_divide(&self, scalar: CoeffType) -> Plaintext {
        let new_poly = self.poly.scalar_integer_divide(scalar, None);
        Plaintext::new(new_poly, self.scaling_factor)
    }

    pub fn multiply(&self, plaintext: &Plaintext) -> Plaintext {
        assert!(self.scaling_factor == plaintext.scaling_factor);
        let new_poly = self
            .poly
            .multiply(&plaintext.poly, self.scaling_factor, None);
        Plaintext::new(new_poly, self.scaling_factor)
    }

    #[cfg(feature = "emulated")]
    pub fn multiply_cipher(&self, ciphertext: &Ciphertext) -> Ciphertext {
        let new_c0 = self
            .poly
            .multiply(&ciphertext.c0, self.scaling_factor, None);
        Ciphertext::new(
            new_c0,
            Polynomial::empty(),
            self.scaling_factor,
            ciphertext.modulus,
        )
    }

    #[cfg(feature = "emulated")]
    pub fn add_cipher(&self, ciphertext: &Ciphertext) -> Ciphertext {
        let new_c0 = self.poly.add(&ciphertext.c0, None);
        Ciphertext::new(
            new_c0,
            Polynomial::empty(),
            self.scaling_factor,
            ciphertext.modulus,
        )
    }

    #[cfg(feature = "emulated")]
    pub fn subtract_cipher(&self, ciphertext: &Ciphertext) -> Ciphertext {
        let new_c0 = self.poly.subtract(&ciphertext.c0, None);
        Ciphertext::new(
            new_c0,
            Polynomial::empty(),
            self.scaling_factor,
            ciphertext.modulus,
        )
    }

    pub fn from_f32_vec_with_scale(values: Vec<f32>, scaling_factor: usize) -> Self {
        let scale = 2f32.powi(scaling_factor as i32);
        let scaled_coeffs: Vec<CoeffType> = values
            .into_iter()
            .map(|v| (v * scale).round() as CoeffType)
            .collect();

        Self {
            poly: Polynomial::new(scaled_coeffs),
            scaling_factor,
        }
    }
}

#[cfg(feature = "emulated")]
impl Plaintext {
    pub fn encrypt(&self) -> Ciphertext {
        Ciphertext::new_emulated(self.poly.clone(), self.scaling_factor)
    }
}

impl From<Vec<f32>> for Plaintext {
    fn from(values: Vec<f32>) -> Self {
        // Use a default scaling factor of 20 (2^20 = 1,048,576)
        Self::from_f32_vec_with_scale(values, 20)
    }
}

impl From<Plaintext> for Vec<f32> {
    fn from(plaintext: Plaintext) -> Vec<f32> {
        let scale = 2f32.powi(plaintext.scaling_factor as i32);
        plaintext
            .poly
            .coeffs
            .iter()
            .map(|&c| c as f32 / scale)
            .collect()
    }
}

#[cfg(test)]
mod plaintext_test {
    #[test]
    fn multiply_two_plaintext() {
        use crate::types::plaintext::Plaintext;

        let p1 = Plaintext::from(vec![1.0, 2.0, 3.0]);
        let p2 = Plaintext::from(vec![4.0, 5.0, 6.0]);

        let result = Vec::from(p1.multiply(&p2));
        println!("{:?}", result);
        assert!(result == vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn add_two_plaintext() {
        use crate::types::plaintext::Plaintext;

        let p1 = Plaintext::from(vec![1.0, 2.0, 3.0]);
        let p2 = Plaintext::from(vec![4.0, 5.0, 6.0]);

        let result = Vec::from(p1.add(&p2));
        println!("{:?}", result);
        assert!(result == vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn subtract_two_plaintext() {
        use crate::types::plaintext::Plaintext;

        let p1 = Plaintext::from(vec![1.0, 2.0, 3.0]);
        let p2 = Plaintext::from(vec![4.0, 5.0, 6.0]);

        let result = Vec::from(p1.subtract(&p2));
        println!("{:?}", result);
        assert!(result == vec![-3.0, -3.0, -3.0]);
    }

    #[test]
    fn multiply_plain_cipher() {
        use crate::types::plaintext::Plaintext;

        let p1 = Plaintext::from(vec![1.0, 2.0, 3.0]);
        let p2 = Plaintext::from(vec![4.0, 5.0, 6.0]);

        let c1 = p1.encrypt();

        let result = Vec::from(p2.multiply_cipher(&c1).decrypt());
        println!("{:?}", result);
        assert!(result == vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn add_plain_cipher() {
        use crate::types::plaintext::Plaintext;

        let p1 = Plaintext::from(vec![1.0, 2.0, 3.0]);
        let p2 = Plaintext::from(vec![4.0, 5.0, 6.0]);

        let c1 = p1.encrypt();

        let result = Vec::from(p2.add_cipher(&c1).decrypt());
        println!("{:?}", result);
        assert!(result == vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn subtract_plain_cipher() {
        use crate::types::plaintext::Plaintext;

        let p1 = Plaintext::from(vec![1.0, 2.0, 3.0]);
        let p2 = Plaintext::from(vec![4.0, 5.0, 6.0]);

        let c2 = p2.encrypt();

        let result = Vec::from(p1.subtract_cipher(&c2).decrypt());
        println!("{:?}", result);
        assert!(result == vec![-3.0, -3.0, -3.0]);
    }
}
