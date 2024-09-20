// ntt_fft.rs

use crate::util::bit_operations::{bit_reverse_vec, reverse_bits};
use num_complex::Complex64;
use std::f64::consts::PI;

pub struct NTTContext {
    coeff_modulus: u64,
    degree: usize,
    roots_of_unity: Vec<u64>,
    roots_of_unity_inv: Vec<u64>,
    reversed_bits: Vec<usize>,
}

impl NTTContext {
    pub fn new(poly_degree: usize, coeff_modulus: u64, root_of_unity: Option<u64>) -> Self {
        assert!(
            poly_degree.is_power_of_two(),
            "Polynomial degree must be a power of 2"
        );

        let mut ctx = NTTContext {
            coeff_modulus,
            degree: poly_degree,
            roots_of_unity: vec![],
            roots_of_unity_inv: vec![],
            reversed_bits: vec![],
        };

        let root = root_of_unity.unwrap_or_else(|| {
            // Implement root_of_unity function from number_theory module
            unimplemented!("root_of_unity function not implemented")
        });

        ctx.precompute_ntt(root);
        ctx
    }

    fn precompute_ntt(&mut self, root_of_unity: u64) {
        self.roots_of_unity = vec![1; self.degree as usize];
        for i in 1..self.degree as usize {
            self.roots_of_unity[i] =
                (self.roots_of_unity[i - 1] * root_of_unity) % self.coeff_modulus;
        }

        let root_of_unity_inv = mod_inv(root_of_unity, self.coeff_modulus);
        self.roots_of_unity_inv = vec![1; self.degree as usize];
        for i in 1..self.degree as usize {
            self.roots_of_unity_inv[i] =
                (self.roots_of_unity_inv[i - 1] * root_of_unity_inv) % self.coeff_modulus;
        }

        let width = self.degree.trailing_zeros() as usize;
        self.reversed_bits = (0..self.degree)
            .map(|i| reverse_bits(i as usize, width) % self.degree)
            .collect();
    }

    pub fn ntt(&self, coeffs: &[u64], rou: &[u64]) -> Vec<u64> {
        let num_coeffs = coeffs.len();
        assert_eq!(
            rou.len(),
            num_coeffs,
            "Length of the roots of unity is too small"
        );

        let mut result = bit_reverse_vec(coeffs);
        let log_num_coeffs = num_coeffs.trailing_zeros() as usize;

        for logm in 1..=log_num_coeffs {
            for j in (0..num_coeffs).step_by(1 << logm) {
                for i in 0..(1 << (logm - 1)) {
                    let index_even = j + i;
                    let index_odd = j + i + (1 << (logm - 1));
                    let rou_idx = i << (1 + log_num_coeffs - logm);
                    let omega_factor = (rou[rou_idx] * result[index_odd]) % self.coeff_modulus;
                    let butterfly_plus = (result[index_even] + omega_factor) % self.coeff_modulus;
                    let butterfly_minus =
                        (result[index_even] - omega_factor).rem_euclid(self.coeff_modulus);
                    result[index_even] = butterfly_plus;
                    result[index_odd] = butterfly_minus;
                }
            }
        }

        result
    }

    pub fn ftt_fwd(&self, coeffs: &[u64]) -> Vec<u64> {
        assert_eq!(
            coeffs.len(),
            self.degree as usize,
            "ftt_fwd: input length does not match context degree"
        );

        let ftt_input: Vec<u64> = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| (c * self.roots_of_unity[i]) % self.coeff_modulus)
            .collect();

        self.ntt(&ftt_input, &self.roots_of_unity)
    }

    pub fn ftt_inv(&self, coeffs: &[u64]) -> Vec<u64> {
        assert_eq!(
            coeffs.len(),
            self.degree as usize,
            "ntt_inv: input length does not match context degree"
        );

        let to_scale_down = self.ntt(coeffs, &self.roots_of_unity_inv);
        let poly_degree_inv = mod_inv(self.degree as u64, self.coeff_modulus);

        to_scale_down
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let temp =
                    (c as u128 * self.roots_of_unity_inv[i] as u128 * poly_degree_inv as u128)
                        % self.coeff_modulus as u128;
                temp as u64
            })
            .collect()
    }
}

pub struct FFTContext {
    fft_length: usize,
    roots_of_unity: Vec<Complex64>,
    roots_of_unity_inv: Vec<Complex64>,
    rot_group: Vec<usize>,
    reversed_bits: Vec<usize>,
}

impl FFTContext {
    pub fn new(fft_length: usize) -> Self {
        let mut ctx = FFTContext {
            fft_length,
            roots_of_unity: vec![],
            roots_of_unity_inv: vec![],
            rot_group: vec![],
            reversed_bits: vec![],
        };
        ctx.precompute_fft();
        ctx
    }

    fn precompute_fft(&mut self) {
        self.roots_of_unity = (0..self.fft_length)
            .map(|i| {
                let angle = 2.0 * PI * i as f64 / self.fft_length as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();

        self.roots_of_unity_inv = (0..self.fft_length)
            .map(|i| {
                let angle = -2.0 * PI * i as f64 / self.fft_length as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();

        let num_slots = self.fft_length / 4;
        let width = num_slots.trailing_zeros() as usize;
        self.reversed_bits = (0..num_slots)
            .map(|i| reverse_bits(i, width) % num_slots)
            .collect();

        self.rot_group = vec![1; num_slots];
        for i in 1..num_slots {
            self.rot_group[i] = (5 * self.rot_group[i - 1]) % self.fft_length;
        }
    }

    pub fn fft(&self, coeffs: Vec<Complex64>, rou: &[Complex64]) -> Vec<Complex64> {
        let num_coeffs = coeffs.len();
        assert!(
            rou.len() >= num_coeffs,
            "Length of the roots of unity is too small"
        );

        let mut result = bit_reverse_vec(&coeffs);
        let log_num_coeffs = num_coeffs.trailing_zeros() as usize;

        for logm in 1..=log_num_coeffs {
            for j in (0..num_coeffs).step_by(1 << logm) {
                for i in 0..(1 << (logm - 1)) {
                    let index_even = j + i;
                    let index_odd = j + i + (1 << (logm - 1));
                    let rou_idx = (i * self.fft_length) >> logm;
                    let omega_factor = rou[rou_idx] * result[index_odd];
                    let butterfly_plus = result[index_even] + omega_factor;
                    let butterfly_minus = result[index_even] - omega_factor;
                    result[index_even] = butterfly_plus;
                    result[index_odd] = butterfly_minus;
                }
            }
        }

        result
    }

    pub fn fft_fwd(&self, coeffs: Vec<Complex64>) -> Vec<Complex64> {
        self.fft(coeffs, &self.roots_of_unity)
    }

    pub fn fft_inv(&self, coeffs: Vec<Complex64>) -> Vec<Complex64> {
        let num_coeffs = coeffs.len();
        let mut result = self.fft(coeffs, &self.roots_of_unity_inv);
        for value in &mut result {
            *value /= num_coeffs as f64;
        }
        result
    }

    pub fn embedding(&self, coeffs: Vec<Complex64>) -> Vec<Complex64> {
        self.check_embedding_input(&coeffs);
        let num_coeffs = coeffs.len();
        let mut result = bit_reverse_vec(&coeffs);
        let log_num_coeffs = num_coeffs.trailing_zeros() as usize;

        for logm in 1..=log_num_coeffs {
            let idx_mod = 1 << (logm + 2);
            let gap = self.fft_length / idx_mod;
            for j in (0..num_coeffs).step_by(1 << logm) {
                for i in 0..(1 << (logm - 1)) {
                    let index_even = j + i;
                    let index_odd = j + i + (1 << (logm - 1));
                    let rou_idx = (self.rot_group[i] % idx_mod) * gap;
                    let omega_factor = self.roots_of_unity[rou_idx] * result[index_odd];
                    let butterfly_plus = result[index_even] + omega_factor;
                    let butterfly_minus = result[index_even] - omega_factor;
                    result[index_even] = butterfly_plus;
                    result[index_odd] = butterfly_minus;
                }
            }
        }

        result
    }

    pub fn embedding_inv(&self, coeffs: Vec<Complex64>) -> Vec<Complex64> {
        self.check_embedding_input(&coeffs);
        let num_coeffs = coeffs.len();
        let mut result = coeffs;
        let log_num_coeffs = num_coeffs.trailing_zeros() as usize;

        for logm in (1..=log_num_coeffs).rev() {
            let idx_mod = 1 << (logm + 2);
            let gap = self.fft_length / idx_mod;
            for j in (0..num_coeffs).step_by(1 << logm) {
                for i in 0..(1 << (logm - 1)) {
                    let index_even = j + i;
                    let index_odd = j + i + (1 << (logm - 1));
                    let rou_idx = (self.rot_group[i] % idx_mod) * gap;
                    let butterfly_plus = result[index_even] + result[index_odd];
                    let butterfly_minus =
                        (result[index_even] - result[index_odd]) * self.roots_of_unity_inv[rou_idx];
                    result[index_even] = butterfly_plus;
                    result[index_odd] = butterfly_minus;
                }
            }
        }

        let mut to_scale_down = bit_reverse_vec(&result);
        for value in &mut to_scale_down {
            *value /= num_coeffs as f64;
        }
        to_scale_down
    }

    fn check_embedding_input(&self, values: &[Complex64]) {
        assert!(
            values.len() <= self.fft_length / 4,
            "Input vector must have length at most {}",
            self.fft_length / 4
        );
    }
}

// Helper function for modular inverse
pub fn mod_inv(a: u64, m: u64) -> u64 {
    assert!(m != 0, "Modulus cannot be zero");
    if m == 1 {
        return 1;
    }

    let mut a = a as i64;
    let m = m as i64;
    let mut m0 = m;
    let mut y = 0;
    let mut x = 1;

    while a > 1 {
        let q = a / m0;
        let t = m0;
        m0 = a % m0;
        a = t;
        let t = y;
        y = x - q * y;
        x = t;
    }

    assert!(a == 1, "Numbers are not coprime");

    if x < 0 {
        x += m;
    }

    x as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_context() {
        let ctx = NTTContext::new(8, 17, Some(2));
        let coeffs = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let ftt_result = ctx.ftt_fwd(&coeffs);
        let inverse_result = ctx.ftt_inv(&ftt_result);
        assert_eq!(coeffs, inverse_result);
    }

    #[test]
    fn test_fft_context() {
        let ctx = FFTContext::new(16);
        let coeffs: Vec<Complex64> = (0..4).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let fft_result = ctx.fft_fwd(coeffs.clone());
        let inverse_result = ctx.fft_inv(fft_result);
        for (a, b) in coeffs.iter().zip(inverse_result.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_embedding() {
        let ctx = FFTContext::new(16);
        let coeffs: Vec<Complex64> = (0..4).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let embedded = ctx.embedding(coeffs.clone());
        let inverse_result = ctx.embedding_inv(embedded);
        for (a, b) in coeffs.iter().zip(inverse_result.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }
}
