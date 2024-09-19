//! A module to perform bit operations.

/// Reverses bits of an integer.
///
/// Reverse bits of the given value with a specified bit width.
///
/// For example, reversing the value 6 = 0b110 with a width of 3
/// would result in 0b011 = 3.
///
/// # Arguments
///
/// * `value` - Value to be reversed.
/// * `width` - Number of bits to consider in reversal.
///
/// # Returns
///
/// The reversed usize value of the input.
pub fn reverse_bits(mut n: usize, width: usize) -> usize {
    let mut rev = 0;
    for _ in 0..width {
        rev = (rev << 1) | (n & 1);
        n >>= 1;
    }
    rev
}

/// Reverses list by reversing the bits of the indices.
///
/// Reverse indices of the given vector.
///
/// For example, reversing the vector [0, 1, 2, 3, 4, 5, 6, 7] would become
/// [0, 4, 2, 6, 1, 5, 3, 7], since 1 = 0b001 reversed is 0b100 = 4,
/// 3 = 0b011 reversed is 0b110 = 6.
///
/// # Arguments
///
/// * `values` - Vector of values to be reversed. Length of vector must be a power of two.
///
/// # Returns
///
/// The reversed vector based on indices.
pub fn bit_reverse_vec<T: Clone>(vec: &[T]) -> Vec<T> {
    let n = vec.len();
    let width = n.trailing_zeros() as usize;
    let mut result = vec![vec[0].clone(); n];
    for i in 0..n {
        result[reverse_bits(i, width)] = vec[i].clone();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(6, 3), 3);
        assert_eq!(reverse_bits(13, 4), 11);
        assert_eq!(reverse_bits(1, 3), 4);
        assert_eq!(reverse_bits(7, 3), 7);
    }

    #[test]
    fn test_bit_reverse_vec() {
        let original = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let expected = vec![0, 4, 2, 6, 1, 5, 3, 7];
        assert_eq!(bit_reverse_vec(&original), expected);

        let original = vec!['a', 'b', 'c', 'd'];
        let expected = vec!['a', 'c', 'b', 'd'];
        assert_eq!(bit_reverse_vec(&original), expected);
    }

    #[test]
    #[should_panic(expected = "Length of input must be a power of two")]
    fn test_bit_reverse_vec_non_power_of_two() {
        let original = vec![0, 1, 2];
        bit_reverse_vec(&original);
    }
}
