use crate::types::polynomial::double_size::DoubleSized;
use crate::types::polynomial::Polynomial;
use crate::util::random_samples::sample_hamming_weight_vector;
use num_traits::{PrimInt, Signed};

pub struct CkksEncryption<T: PrimInt + Signed + DoubleSized, const N: usize> {
    pub(super) secret_key: Polynomial<T, N>,
    pub(super) modulus: T::Double,
}

impl<T, const N: usize> CkksEncryption<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    pub fn new(modulus: T::Double, hamming_weight: usize) -> Self {
        let secret_key = generate_secret_key::<T, N>(hamming_weight);
        Self {
            secret_key,
            modulus,
        }
    }
}

/// Generates a secret key for CKKS scheme.
/// Args:
///     params (Parameters): Parameters including polynomial degree,
///         plaintext, and ciphertext modulus.
fn generate_secret_key<T: PrimInt + Signed + DoubleSized, const N: usize>(
    weight: usize,
) -> Polynomial<T, N> {
    let key = sample_hamming_weight_vector(N, weight);
    Polynomial::new(key)
}

#[cfg(test)]
mod key_generator_tests {
    use super::*;

    #[test]
    fn test_generate_secret_key() {
        const N: usize = 16;
        let weight = 4;
        let secret_key = generate_secret_key::<i32, N>(weight);
        assert_eq!(secret_key.coeffs.len(), N);
        assert_eq!(
            secret_key
                .coeffs
                .iter()
                .filter(|&&x| x.value() != 0)
                .count(),
            weight
        );
    }

    #[test]
    fn test_generate_secret_key_zero_weight() {
        const N: usize = 16;
        let weight = 0;
        let secret_key = generate_secret_key::<i32, N>(weight);
        assert_eq!(secret_key.coeffs.len(), N);
        assert_eq!(
            secret_key
                .coeffs
                .iter()
                .filter(|&&x| x.value() != 0)
                .count(),
            weight
        );
    }

    #[test]
    fn test_generate_secret_key_full_weight() {
        const N: usize = 16;
        let weight = N;
        let secret_key = generate_secret_key::<i32, N>(weight);
        assert_eq!(secret_key.coeffs.len(), N);
        assert_eq!(
            secret_key
                .coeffs
                .iter()
                .filter(|&&x| x.value() != 0)
                .count(),
            weight
        );
    }

    #[test]
    fn test_generate_secret_key_different_type() {
        const N: usize = 16;
        let weight = 4;
        let secret_key = generate_secret_key::<i64, N>(weight);
        assert_eq!(secret_key.coeffs.len(), N);
        assert_eq!(
            secret_key
                .coeffs
                .iter()
                .filter(|&&x| x.value() != 0)
                .count(),
            weight
        );
    }
}
