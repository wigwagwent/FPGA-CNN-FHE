use log::debug;
use num_traits::{PrimInt, Signed};

use crate::types::ciphertext::Ciphertext;
use crate::types::polynomial::double_size::DoubleSized;
use crate::types::polynomial::Polynomial;

/// A plaintext polynomial. This is a polynomial with integer coefficients that is used to represent
/// the plaintext data. The coefficients of the polynomial are typically small integers. The plaintext
/// polynomial is encrypted to produce a ciphertext polynomial.
#[derive(Clone, Debug, PartialEq)]
pub struct Plaintext<T, const N: usize>
where
    T: PrimInt + Signed + Clone + DoubleSized,
{
    /// The polynomial representing the plaintext data.
    pub(crate) poly: Polynomial<T, N>,
    /// The scaling factor of the plaintext. 2^scaling_factor is the factor by which the plaintext
    /// was scaled.
    pub(crate) scaling_factor: usize,
}

impl<T, const N: usize> Plaintext<T, N>
where
    T: PrimInt + Signed + Clone + DoubleSized,
{
    pub fn new(poly: Polynomial<T, N>, scaling_factor: usize) -> Self {
        #[cfg(feature = "debug")]
        if !N.is_power_of_two() {
            debug!("N is not a power of 2");
        }

        Self {
            poly,
            scaling_factor,
        }
    }

    pub fn from_f32(values: Vec<f32>, scaling_factor: usize) -> Self {
        #[cfg(feature = "debug")]
        if !N.is_power_of_two() {
            debug!("N is not a power of 2");
        }

        let scale = 2f32.powi(scaling_factor as i32);
        let scaled_coeffs: Vec<T> = values
            .into_iter()
            .map(|v| {
                T::from((v * scale).round() as i64)
                    .expect("Input value is too large on plaintext::from_f32")
            })
            .collect::<Vec<T>>();

        Self {
            poly: Polynomial::new(scaled_coeffs),
            scaling_factor,
        }
    }
}

impl<T, const N: usize> Plaintext<T, N>
where
    T: PrimInt + Signed + Clone + DoubleSized,
{
    pub fn len(&self) -> usize {
        N
    }

    pub fn used_len(&self) -> usize {
        self.poly.used_len()
    }

    pub fn add(&self, plaintext: &Plaintext<T, N>) -> Plaintext<T, N> {
        assert!(self.scaling_factor == plaintext.scaling_factor);
        let new_poly = self.poly.add(&plaintext.poly, None);
        Plaintext::new(new_poly, self.scaling_factor)
    }

    pub fn subtract(&self, plaintext: &Plaintext<T, N>) -> Plaintext<T, N> {
        assert!(self.scaling_factor == plaintext.scaling_factor);
        let new_poly = self.poly.subtract(&plaintext.poly, None);
        Plaintext::new(new_poly, self.scaling_factor)
    }

    pub fn multiply(&self, plaintext: &Plaintext<T, N>) -> Plaintext<T, N> {
        assert!(self.scaling_factor == plaintext.scaling_factor);
        let new_poly = self
            .poly
            .multiply(&plaintext.poly, self.scaling_factor, None);
        Plaintext::new(new_poly, self.scaling_factor)
    }

    pub fn rotate(&self, steps: usize) -> Plaintext<T, N> {
        let new_poly = self.poly.rotate_right(steps);
        Plaintext::new(new_poly, self.scaling_factor)
    }
}

#[cfg(feature = "emulated")]
impl<T, const N: usize> Plaintext<T, N>
where
    T: PrimInt + Signed + Clone + DoubleSized,
{
    pub fn encrypt(&self) -> Ciphertext<T, N> {
        Ciphertext::new_emulated(self.poly.clone(), self.scaling_factor)
    }
}

impl<T, const N: usize> From<Plaintext<T, N>> for Vec<f32>
where
    T: PrimInt + Signed + Clone + DoubleSized,
{
    fn from(plaintext: Plaintext<T, N>) -> Vec<f32> {
        let scale = 2f32.powi(plaintext.scaling_factor as i32);
        plaintext
            .poly
            .coeffs
            .iter()
            .map(|c| T::to_f32(&c.value()).unwrap() / scale)
            .collect()
    }
}

#[cfg(test)]
mod plaintext_test {
    #[test]
    fn multiply_two_plaintext() {
        use crate::types::plaintext::Plaintext;

        let p1: Plaintext<i32, 3> = Plaintext::from_f32(vec![1.0, 2.0, 3.0], 15);
        let p2: Plaintext<i32, 3> = Plaintext::from_f32(vec![4.0, 5.0, 6.0], 15);

        let result = Vec::from(p1.multiply(&p2));
        println!("{:?}", result);
        assert!(result == vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn add_two_plaintext() {
        use crate::types::plaintext::Plaintext;

        let p1: Plaintext<i32, 3> = Plaintext::from_f32(vec![1.0, 2.0, 3.0], 15);
        let p2: Plaintext<i32, 3> = Plaintext::from_f32(vec![4.0, 5.0, 6.0], 15);

        let result = Vec::from(p1.add(&p2));
        println!("{:?}", result);
        assert!(result == vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn subtract_two_plaintext() {
        use crate::types::plaintext::Plaintext;

        let p1: Plaintext<i32, 3> = Plaintext::from_f32(vec![1.0, 2.0, 3.0], 15);
        let p2: Plaintext<i32, 3> = Plaintext::from_f32(vec![4.0, 5.0, 6.0], 15);

        let result = Vec::from(p1.subtract(&p2));
        println!("{:?}", result);
        assert!(result == vec![-3.0, -3.0, -3.0]);
    }
}
