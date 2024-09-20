use num_traits::{PrimInt, Signed};
use rand::Rng;

use crate::DoubleSized;

/// Samples from a Hamming weight distribution.
///
/// Samples uniformly from the set [-1, 0, 1] such that the
/// resulting vector has exactly h nonzero values.
///
///  Args:
///     length (int): Length of resulting vector.
///      hamming_weight (int): Hamming weight h of resulting vector.
/// Returns:
///     A list of randomly sampled values.
pub fn sample_hamming_weight_vector<T>(length: usize, hamming_weight: usize) -> Vec<T>
where
    T: PrimInt + Signed + DoubleSized,
{
    assert!(hamming_weight <= length);
    let mut sample = vec![T::zero(); length];
    let mut total_weight = 0;

    while total_weight < hamming_weight {
        let index = rand::thread_rng().gen_range(0..length);
        if sample[index] == T::zero() {
            let r = rand::thread_rng().gen_bool(0.5);
            if r {
                sample[index] = T::from(-1).unwrap();
            } else {
                sample[index] = T::one();
            }
            total_weight += 1;
        }
    }

    sample
}

/// Samples from a discrete triangle distribution.
/// Samples num_samples values from [-1, 0, 1] with probabilities
/// [0.25, 0.5, 0.25], respectively.
/// Args:
///     num_samples (int): Number of samples to be drawn.
/// Returns:
///     A list of randomly sampled values.
pub fn sample_triangle<T>(num_samples: usize) -> Vec<T>
where
    T: PrimInt + Signed + DoubleSized,
{
    let mut sample: Vec<T> = vec![T::zero(); num_samples];

    for i in 0..num_samples {
        let r = rand::thread_rng().gen_range(0..4);
        if r == 0 {
            sample[i] = T::from(-1).unwrap();
        } else if r == 1 {
            sample[i] = T::one();
        }
    }

    sample
}

#[cfg(test)]
mod random_samples_tests {
    use super::*;

    #[test]
    fn test_sample_hamming_weight_vector() {
        let length = 10;
        let hamming_weight = 3;
        let sample: Vec<i32> = sample_hamming_weight_vector(length, hamming_weight);

        assert_eq!(sample.len(), length);
        assert_eq!(sample.iter().filter(|&&x| x != 0).count(), hamming_weight);
        assert!(sample.iter().all(|&x| x == -1 || x == 0 || x == 1));
    }

    #[test]
    fn test_sample_hamming_weight_vector_zero_weight() {
        let length = 10;
        let hamming_weight = 0;
        let sample: Vec<i32> = sample_hamming_weight_vector(length, hamming_weight);

        assert_eq!(sample.len(), length);
        assert_eq!(sample.iter().filter(|&&x| x != 0).count(), hamming_weight);
        assert!(sample.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_sample_hamming_weight_vector_full_weight() {
        let length = 10;
        let hamming_weight = 10;
        let sample: Vec<i32> = sample_hamming_weight_vector(length, hamming_weight);

        assert_eq!(sample.len(), length);
        assert_eq!(sample.iter().filter(|&&x| x != 0).count(), hamming_weight);
        assert!(sample.iter().all(|&x| x == -1 || x == 1));
    }

    #[test]
    fn test_sample_hamming_weight_vector_large_length() {
        let length = 1000;
        let hamming_weight = 500;
        let sample: Vec<i32> = sample_hamming_weight_vector(length, hamming_weight);

        assert_eq!(sample.len(), length);
        assert_eq!(sample.iter().filter(|&&x| x != 0).count(), hamming_weight);
        assert!(sample.iter().all(|&x| x == -1 || x == 0 || x == 1));
    }

    #[test]
    #[should_panic]
    fn test_sample_hamming_weight_vector_invalid_weight() {
        let length = 10;
        let hamming_weight = 11;
        sample_hamming_weight_vector(length, hamming_weight) as Vec<i32>;
    }
}
