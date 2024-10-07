use num_traits::{PrimInt, Signed};

use crate::types::ciphertext::Ciphertext;
use crate::types::plaintext::Plaintext;
use crate::types::polynomial::double_size::DoubleSized;
use crate::types::polynomial::Polynomial;
use crate::CkksEncryption;

impl<T: PrimInt + Signed + DoubleSized, const N: usize> CkksEncryption<T, N> {
    /// Decrypts a ciphertext.
    /// Decrypts the ciphertext and returns the corresponding plaintext.
    /// Args:
    ///     ciphertext (Ciphertext): Ciphertext to be decrypted.
    ///     c2 (Polynomial): Optional additional parameter for a ciphertext that
    ///         has not been relinearized.
    /// Returns:
    ///     The plaintext corresponding to the decrypted ciphertext.
    pub fn decrypt(
        &self,
        ciphertext: Ciphertext<T, N>,
        c2: Option<Polynomial<T, N>>,
    ) -> Plaintext<T, N> {
        let message = ciphertext
            .c1
            .multiply(&self.secret_key, 0, Some(ciphertext.modulus));
        let message = ciphertext.c0.add(&message, Some(ciphertext.modulus));
        if c2.is_some() {
            todo!()
            // secret_key_squared = self.secret_key.s.multiply(self.secret_key.s, ciphertext.modulus)
            //             c2_message = c2.multiply(secret_key_squared, ciphertext.modulus, crt=self.crt_context)
            //             message = message.add(c2_message, ciphertext.modulus)
        }

        let message = message.mod_small(Some(ciphertext.modulus));
        Plaintext {
            poly: message,
            scaling_factor: ciphertext.scaling_factor,
        }
    }
}

#[cfg(test)]
mod encrypter_tests {
    use super::*;

    #[test]
    fn test_encrypt_then_decrypt() {
        let encrypter: CkksEncryption<i16, 1024> = CkksEncryption::new(1 << 30, 800);

        let plain = Plaintext::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2);
        let ciphertext = encrypter.encrypt_with_secret_key(plain.clone());

        let decrypted = encrypter.decrypt(ciphertext, None);
        assert_eq!(decrypted, plain);
    }
}
