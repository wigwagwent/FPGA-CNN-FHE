use num_traits::{PrimInt, Signed};

use crate::types::plaintext::Plaintext;
use crate::types::polynomial::Polynomial;

use super::polynomial::double_size::DoubleSized;

#[derive(Clone, Debug, PartialEq)]
pub struct Ciphertext<T, const N: usize>
where
    T: PrimInt + Signed + Clone + DoubleSized,
{
    /// The first polynomial in the ciphertext. This polynomial is used to store the result of the
    /// homomorphic operations. The coefficients of this polynomial are typically larger than the
    /// coefficients of the second polynomial.
    pub(crate) c0: Polynomial<T, N>,
    /// The second polynomial in the ciphertext. This polynomial is used to store the noise in the
    /// ciphertext. The coefficients of this polynomial are typically smaller than the coefficients
    /// of the first polynomial. The noise in the ciphertext grows as homomorphic operations are
    /// performed on the ciphertext. The noise budget of the ciphertext is proportional to the
    /// scaling factor.
    pub(crate) c1: Polynomial<T, N>,
    /// The scaling factor of the ciphertext. 2^scaling_factor is the factor by which the plaintext
    /// was scaled before encryption. This is used to determine the noise budget of the ciphertext.
    /// The noise budget is the number of bits of noise present in the ciphertext, which is
    /// proportional to the scaling factor.
    pub(crate) scaling_factor: usize,
    /// The modulus of the polynomial coefficients. All operations on the ciphertext are performed
    /// modulo this value. This is typically a prime number. The modulus is used to ensure that the
    /// polynomial coefficients do not grow too large during homomorphic operations. The modulus
    /// should be chosen such that the coefficients do not overflow when performing homomorphic operations.
    pub(crate) modulus: T::Double,
}

impl<T, const N: usize> Ciphertext<T, N>
where
    T: PrimInt + Signed + Clone + DoubleSized,
{
    pub fn new(
        c0: Polynomial<T, N>,
        c1: Polynomial<T, N>,
        scaling_factor: usize,
        modulus: i128,
    ) -> Self {
        Self {
            c0,
            c1,
            scaling_factor,
            modulus: T::from(modulus).unwrap().to_double(),
        }
    }
}

impl<T, const N: usize> Ciphertext<T, N>
where
    T: PrimInt + Signed + Clone + DoubleSized,
{
    pub fn len(&self) -> usize {
        N
    }

    pub fn used_len(&self) -> usize {
        self.c0.used_len()
    }
}

#[cfg(feature = "emulated")]
impl<T, const N: usize> Ciphertext<T, N>
where
    T: PrimInt + Signed + Clone + DoubleSized,
{
    pub fn decrypt(&self) -> Plaintext<T, N> {
        Plaintext::new(self.c0.clone(), self.scaling_factor)
    }

    pub fn new_emulated(c0: Polynomial<T, N>, scaling_factor: usize) -> Self {
        Self {
            c0,
            c1: Polynomial::default(),
            scaling_factor,
            modulus: T::from(1).unwrap().to_double(),
        }
    }

    pub fn add(&self, ciphertext: &Ciphertext<T, N>) -> Ciphertext<T, N> {
        assert!(self.scaling_factor == ciphertext.scaling_factor);
        let new_c0 = self.c0.add(&ciphertext.c0, None);
        Ciphertext::new(new_c0, Polynomial::default(), self.scaling_factor, 1)
    }

    pub fn add_plain(&self, plaintext: &Plaintext<T, N>) -> Ciphertext<T, N> {
        assert!(self.scaling_factor == plaintext.scaling_factor);
        let new_c0 = self.c0.add(&plaintext.poly, None);
        Ciphertext::new(new_c0, Polynomial::default(), self.scaling_factor, 1)
    }

    pub fn subtract(&self, ciphertext: &Ciphertext<T, N>) -> Ciphertext<T, N> {
        assert!(self.scaling_factor == ciphertext.scaling_factor);
        let new_c0 = self.c0.subtract(&ciphertext.c0, None);
        Ciphertext::new(new_c0, Polynomial::default(), self.scaling_factor, 1)
    }

    pub fn subtract_plain(&self, plaintext: &Plaintext<T, N>) -> Ciphertext<T, N> {
        assert!(self.scaling_factor == plaintext.scaling_factor);
        let new_c0 = self.c0.subtract(&plaintext.poly, None);
        Ciphertext::new(new_c0, Polynomial::default(), self.scaling_factor, 1)
    }

    pub fn multiply(&self, ciphertext: &Ciphertext<T, N>) -> Ciphertext<T, N> {
        assert!(self.scaling_factor == ciphertext.scaling_factor);
        let new_c0 = self.c0.multiply(&ciphertext.c0, self.scaling_factor, None);
        Ciphertext::new(new_c0, Polynomial::default(), self.scaling_factor, 1)
    }

    pub fn multiply_plain(&self, plaintext: &Plaintext<T, N>) -> Ciphertext<T, N> {
        assert!(self.scaling_factor == plaintext.scaling_factor);
        let new_c0 = self.c0.multiply(&plaintext.poly, self.scaling_factor, None);
        Ciphertext::new(new_c0, Polynomial::default(), self.scaling_factor, 1)
    }
}

#[cfg(test)]
mod ciphertext_test {
    use crate::Ciphertext;

    #[test]
    fn multiply_two_ciphertext() {
        use crate::types::plaintext::Plaintext;

        let p1: Plaintext<i32, 3> = Plaintext::from_f32(vec![1.0, 2.0, 3.0], 15);
        let p2: Plaintext<i32, 3> = Plaintext::from_f32(vec![4.0, 5.0, 6.0], 15);

        let c1: Ciphertext<i32, 3> = p1.encrypt();
        let c2: Ciphertext<i32, 3> = p2.encrypt();

        let c3: Ciphertext<i32, 3> = c1.multiply(&c2);
        let result = Vec::from(c3.decrypt());
        println!("{:?}", result);
        assert_eq!(result, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn add_two_ciphertext() {
        use crate::types::plaintext::Plaintext;

        let p1: Plaintext<i32, 3> = Plaintext::from_f32(vec![1.0, 2.0, 3.0], 15);
        let p2: Plaintext<i32, 3> = Plaintext::from_f32(vec![4.0, 5.0, 6.0], 15);

        let c1: Ciphertext<i32, 3> = p1.encrypt();
        let c2: Ciphertext<i32, 3> = p2.encrypt();

        let c3: Ciphertext<i32, 3> = c1.add(&c2);
        let result = Vec::from(c3.decrypt());
        println!("{:?}", result);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn subtract_two_ciphertext() {
        use crate::types::plaintext::Plaintext;

        let p1: Plaintext<i32, 3> = Plaintext::from_f32(vec![1.0, 2.0, 3.0], 15);
        let p2: Plaintext<i32, 3> = Plaintext::from_f32(vec![4.0, 5.0, 6.0], 15);

        let c1: Ciphertext<i32, 3> = p1.encrypt();
        let c2: Ciphertext<i32, 3> = p2.encrypt();

        let c3: Ciphertext<i32, 3> = c1.subtract(&c2);
        let result = Vec::from(c3.decrypt());
        println!("{:?}", result);
        assert_eq!(result, vec![-3.0, -3.0, -3.0]);
    }

    #[test]
    fn multiply_cipher_plain() {
        use crate::types::plaintext::Plaintext;

        let p1: Plaintext<i32, 3> = Plaintext::from_f32(vec![1.0, 2.0, 3.0], 15);
        let p2: Plaintext<i32, 3> = Plaintext::from_f32(vec![4.0, 5.0, 6.0], 15);

        let c1: Ciphertext<i32, 3> = p1.encrypt();

        let c2: Ciphertext<i32, 3> = c1.multiply_plain(&p2);
        let result = Vec::from(c2.decrypt());
        println!("{:?}", result);
        assert_eq!(result, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn add_cipher_plain() {
        use crate::types::plaintext::Plaintext;

        let p1: Plaintext<i32, 3> = Plaintext::from_f32(vec![1.0, 2.0, 3.0], 15);
        let p2: Plaintext<i32, 3> = Plaintext::from_f32(vec![4.0, 5.0, 6.0], 15);

        let c1: Ciphertext<i32, 3> = p1.encrypt();

        let c2: Ciphertext<i32, 3> = c1.add_plain(&p2);
        let result = Vec::from(c2.decrypt());
        println!("{:?}", result);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn subtract_cipher_plain() {
        use crate::types::plaintext::Plaintext;

        let p1: Plaintext<i32, 3> = Plaintext::from_f32(vec![1.0, 2.0, 3.0], 15);
        let p2: Plaintext<i32, 3> = Plaintext::from_f32(vec![4.0, 5.0, 6.0], 15);

        let c2: Ciphertext<i32, 3> = p2.encrypt();

        let c1: Ciphertext<i32, 3> = c2.subtract_plain(&p1);
        let result = Vec::from(c1.decrypt());
        println!("{:?}", result);
        assert_eq!(result, vec![3.0, 3.0, 3.0]);
    }
}
