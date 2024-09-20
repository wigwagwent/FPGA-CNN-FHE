use num_traits::{PrimInt, Signed};

use crate::{
    util::random_samples::sample_triangle, Ciphertext, CkksEncryption, DoubleSized, Plaintext,
    Polynomial,
};

/// Encrypts a message with secret key encryption.
/// Encrypts the message for secret key encryption and returns the corresponding ciphertext.
/// Args:
///     plain (Plaintext): Plaintext to be encrypted.
/// Returns:
///     A ciphertext consisting of a pair of polynomials in the ciphertext
///     space.
impl<T: PrimInt + Signed + DoubleSized, const N: usize> CkksEncryption<T, N> {
    pub fn encrypt_with_secret_key(&self, plain: Plaintext<T, N>) -> Ciphertext<T, N> {
        let random_vec = Polynomial::new(sample_triangle(N));
        let error = Polynomial::new(sample_triangle(N));

        let c0 = self
            .secret_key
            .multiply(&random_vec, 0, None)
            .add(&error, None)
            .add(&plain.poly, None);
        let c0 = c0.mod_small(None);

        let c1 = random_vec
            .scalar_multiply(T::from(-1).unwrap(), None)
            .mod_small(None);

        Ciphertext::new(c0, c1, plain.scaling_factor, self.modulus)
    }
}

#[cfg(test)]
mod encrypter_tests {
    use super::*;

    #[test]
    fn test_encrypt_with_secret_key_nonzero() {
        let encrypter: CkksEncryption<i16, 1024> = CkksEncryption::new(1 << 30, 800);
        let plain = Plaintext::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2);
        let ciphertext = encrypter.encrypt_with_secret_key(plain);

        // Ensure that the ciphertext is not zero
        assert!(ciphertext.c0.used_len() > 0);
    }
}
