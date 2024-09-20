use crate::util::crt::CRTContext;
use crate::util::ntt::{FFTContext, NTTContext};

#[derive(Clone)]
pub struct Polynomial {
    ring_degree: usize,
    coeffs: Vec<u64>,
}

impl Polynomial {
    pub fn new(degree: usize, coeffs: Vec<u64>) -> Self {
        assert_eq!(
            coeffs.len(),
            degree,
            "Size of polynomial array {} is not equal to degree {} of ring",
            coeffs.len(),
            degree
        );
        Self {
            ring_degree: degree,
            coeffs,
        }
    }

    pub fn add(&self, poly: &Polynomial, coeff_modulus: Option<usize>) -> Polynomial {
        let mut poly_sum = vec![0; self.ring_degree];
        for i in 0..self.ring_degree {
            poly_sum[i] = self.coeffs[i] + poly.coeffs[i];
        }
        let mut result = Polynomial::new(self.ring_degree, poly_sum);
        if let Some(modulus) = coeff_modulus {
            result = result.mod_small(modulus);
        }
        result
    }

    pub fn subtract(&self, poly: &Polynomial, coeff_modulus: Option<usize>) -> Polynomial {
        let mut poly_diff = vec![0; self.ring_degree];
        for i in 0..self.ring_degree {
            poly_diff[i] = self.coeffs[i] - poly.coeffs[i];
        }
        let mut result = Polynomial::new(self.ring_degree, poly_diff);
        if let Some(modulus) = coeff_modulus {
            result = result.mod_small(modulus);
        }
        result
    }

    pub fn multiply(
        &self,
        poly: &Polynomial,
        coeff_modulus: u64,
        ntt: Option<&NTTContext>,
        crt: Option<&CRTContext>,
    ) -> Polynomial {
        if let Some(crt_context) = crt {
            return self.multiply_crt(poly, crt_context);
        }
        if let Some(ntt_context) = ntt {
            let a = ntt_context.ftt_fwd(&self.coeffs);
            let b = ntt_context.ftt_fwd(&poly.coeffs);
            let ab: Vec<u64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
            let prod = ntt_context.ftt_inv(&ab);
            return Polynomial::new(self.ring_degree, prod);
        }
        self.multiply_naive(poly, Some(coeff_modulus))
    }

    pub fn multiply_crt(&self, poly: &Polynomial, crt: &CRTContext) -> Polynomial {
        let mut poly_prods = Vec::new();
        for i in 0..crt.primes.len() {
            let prod = self.multiply(poly, crt.primes[i], Some(&crt.ntts[i]), None);
            poly_prods.push(prod);
        }
        let mut final_coeffs: Vec<i64> = vec![0; self.ring_degree as i64];
        for i in 0..self.ring_degree {
            let values: Vec<i64> = poly_prods.iter().map(|p| p.coeffs[i]).collect();
            final_coeffs[i] = crt.reconstruct(&values);
        }
        Polynomial::new(self.ring_degree, final_coeffs).mod_small(crt.modulus)
    }

    pub fn multiply_fft(&self, poly: &Polynomial, round: bool) -> Polynomial {
        let fft = FFTContext::new(self.ring_degree * 8);
        let mut a = self.coeffs.clone();
        a.extend(vec![0; self.ring_degree]);
        let mut b = poly.coeffs.clone();
        b.extend(vec![0; self.ring_degree]);
        let a = fft.fft_fwd(&a);
        let b = fft.fft_fwd(&b);
        let ab: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        let prod = fft.fft_inv(&ab);
        let mut poly_prod = vec![0.0; self.ring_degree];
        for d in 0..(2 * self.ring_degree - 1) {
            let index = d % self.ring_degree;
            let sign = if d < self.ring_degree { 1.0 } else { -1.0 };
            poly_prod[index] += sign * prod[d];
        }
        let result = Polynomial::new(
            self.ring_degree,
            poly_prod.iter().map(|&x| x as i64).collect(),
        );
        if round {
            result.round()
        } else {
            result
        }
    }

    pub fn multiply_naive(&self, poly: &Polynomial, coeff_modulus: Option<u64>) -> Polynomial {
        let mut poly_prod = vec![0i64; self.ring_degree];
        let n = self.ring_degree;

        for d in 0..(2 * n - 1) {
            let index = d % n;
            let is_wrapped = d >= n;
            let sign = if is_wrapped { -1i64 } else { 1i64 };

            let mut coeff = 0i64;
            for i in 0..n {
                if let Some(j) = d.checked_sub(i) {
                    if j < n {
                        coeff =
                            coeff.wrapping_add(self.coeffs[i].wrapping_mul(poly.coeffs[j]) as i64);
                    }
                }
            }

            poly_prod[index] = poly_prod[index].wrapping_add(sign.wrapping_mul(coeff));

            if let Some(modulus) = coeff_modulus {
                poly_prod[index] = mod_i64(poly_prod[index], modulus);
            }
        }

        Polynomial::new(n, poly_prod)
    }

    pub fn scalar_multiply(&self, scalar: i64, coeff_modulus: Option<i64>) -> Polynomial {
        let new_coeffs = if let Some(modulus) = coeff_modulus {
            self.coeffs
                .iter()
                .map(|&c| (scalar * c) % modulus)
                .collect()
        } else {
            self.coeffs.iter().map(|&c| scalar * c).collect()
        };
        Polynomial::new(self.ring_degree, new_coeffs)
    }

    pub fn scalar_integer_divide(&self, scalar: i64, coeff_modulus: Option<i64>) -> Polynomial {
        let new_coeffs = if let Some(modulus) = coeff_modulus {
            self.coeffs
                .iter()
                .map(|&c| ((c / scalar) % modulus + modulus) % modulus)
                .collect()
        } else {
            self.coeffs.iter().map(|&c| c / scalar).collect()
        };
        Polynomial::new(self.ring_degree, new_coeffs)
    }

    pub fn rotate(&self, r: usize) -> Polynomial {
        let k = 5_i64.pow(r as u32) as usize;
        let mut new_coeffs = vec![0; self.ring_degree];
        for i in 0..self.ring_degree {
            let index = (i * k) % (2 * self.ring_degree);
            if index < self.ring_degree {
                new_coeffs[index] = self.coeffs[i];
            } else {
                new_coeffs[index - self.ring_degree] = -self.coeffs[i];
            }
        }
        Polynomial::new(self.ring_degree, new_coeffs)
    }

    pub fn conjugate(&self) -> Polynomial {
        let mut new_coeffs = vec![0; self.ring_degree];
        new_coeffs[0] = self.coeffs[0];
        for i in 1..self.ring_degree {
            new_coeffs[i] = -self.coeffs[self.ring_degree - i];
        }
        Polynomial::new(self.ring_degree, new_coeffs)
    }

    pub fn round(&self) -> Polynomial {
        // For integers, this is a no-op
        Polynomial::new(self.ring_degree, self.coeffs.clone())
    }

    pub fn floor(&self) -> Polynomial {
        // For integers, this is a no-op
        Polynomial::new(self.ring_degree, self.coeffs.clone())
    }

    pub fn mod_small(&self, coeff_modulus: usize) -> Polynomial {
        let new_coeffs: Vec<i64> = self
            .coeffs
            .iter()
            .map(|&c| {
                let mut c = c % (coeff_modulus as i64);
                if c > (coeff_modulus as i64) / 2 {
                    c -= coeff_modulus as i64;
                } else if c < -(coeff_modulus as i64) / 2 {
                    c += coeff_modulus as i64;
                }
                c
            })
            .collect();

        Polynomial::new(self.ring_degree, new_coeffs)
    }

    pub fn base_decompose(&self, base: i64, num_levels: usize) -> Vec<Polynomial> {
        let mut decomposed =
            vec![Polynomial::new(self.ring_degree, vec![0; self.ring_degree]); num_levels];
        let mut poly = self.clone();
        for i in 0..num_levels {
            decomposed[i] = poly.mod_small(base);
            poly = poly.scalar_multiply(1, Some(base)).floor();
        }
        decomposed
    }

    pub fn evaluate(&self, inp: i64) -> i64 {
        let mut result = self.coeffs[self.ring_degree - 1];
        for i in (0..self.ring_degree - 1).rev() {
            result = result * inp + self.coeffs[i];
        }
        result
    }
}

// Helper function to perform modulo operation on i64 with u64 modulus
fn mod_i64(a: i64, m: u64) -> i64 {
    let m = m as i64;
    ((a % m + m) % m) as i64
}

impl std::fmt::Display for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in (0..self.ring_degree).rev() {
            if self.coeffs[i] != 0 {
                if !s.is_empty() {
                    s.push_str(" + ");
                }
                if i == 0 || self.coeffs[i] != 1 {
                    s.push_str(&self.coeffs[i].to_string());
                }
                if i != 0 {
                    s.push('x');
                }
                if i > 1 {
                    s.push_str(&format!("^{}", i));
                }
            }
        }
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let p1 = Polynomial::new(4, vec![1, 2, 3, 4]);
        let p2 = Polynomial::new(4, vec![5, 6, 7, 8]);
        let result = p1.add(&p2, Some(11));
        assert_eq!(result.coeffs, vec![6, 8, 10, 1]);
    }

    #[test]
    fn test_subtract() {
        let p1 = Polynomial::new(4, vec![5, 6, 7, 8]);
        let p2 = Polynomial::new(4, vec![1, 2, 3, 4]);
        let result = p1.subtract(&p2, Some(11));
        assert_eq!(result.coeffs, vec![4, 4, 4, 4]);
    }

    #[test]
    fn test_scalar_multiply() {
        let p = Polynomial::new(4, vec![1, 2, 3, 4]);
        let result = p.scalar_multiply(3, Some(11));
        assert_eq!(result.coeffs, vec![3, 6, 9, 1]);
    }

    #[test]
    fn test_rotate() {
        let p = Polynomial::new(4, vec![1, 2, 3, 4]);
        let result = p.rotate(1);
        assert_eq!(result.coeffs, vec![4, -1, -2, -3]);
    }

    #[test]
    fn test_conjugate() {
        let p = Polynomial::new(4, vec![1, 2, 3, 4]);
        let result = p.conjugate();
        assert_eq!(result.coeffs, vec![1, -4, -3, -2]);
    }
}
