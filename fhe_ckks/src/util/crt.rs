use crate::util::ntt::NTTContext;

use super::ntt::mod_inv;

pub struct CRTContext {
    poly_degree: usize,
    pub primes: Vec<u64>,
    pub modulus: u128,
    pub ntts: Vec<NTTContext>,
    crt_vals: Vec<u64>,
    crt_inv_vals: Vec<u64>,
}

impl CRTContext {
    pub fn new(num_primes: usize, prime_size: u64, poly_degree: usize) -> Self {
        let mut ctx = CRTContext {
            poly_degree,
            primes: Vec::new(),
            modulus: 1,
            ntts: Vec::new(),
            crt_vals: Vec::new(),
            crt_inv_vals: Vec::new(),
        };

        ctx.generate_primes(num_primes, prime_size, 2 * poly_degree as u64);
        ctx.generate_ntt_contexts();

        for &prime in &ctx.primes {
            ctx.modulus *= prime as u128;
        }

        ctx.precompute_crt();
        ctx
    }

    fn generate_primes(&mut self, num_primes: usize, prime_size: u64, mod_val: u64) {
        let mut possible_prime = (1 << prime_size) + 1;

        for _ in 0..num_primes {
            possible_prime += mod_val;
            while !is_prime(possible_prime) {
                possible_prime += mod_val;
            }
            self.primes.push(possible_prime);
        }
    }

    fn generate_ntt_contexts(&mut self) {
        for &prime in &self.primes {
            let ntt = NTTContext::new(self.poly_degree, prime, None);
            self.ntts.push(ntt);
        }
    }

    fn precompute_crt(&mut self) {
        let num_primes = self.primes.len();
        self.crt_vals = vec![1; num_primes];
        self.crt_inv_vals = vec![1; num_primes];

        for i in 0..num_primes {
            self.crt_vals[i] = (self.modulus / self.primes[i] as u128) as u64;
            self.crt_inv_vals[i] = mod_inv(self.crt_vals[i], self.primes[i]);
        }
    }

    pub fn crt(&self, value: u64) -> Vec<u64> {
        self.primes.iter().map(|&p| value % p).collect()
    }

    pub fn reconstruct(&self, values: &[u64]) -> u64 {
        assert_eq!(values.len(), self.primes.len());

        let mut regular_rep_val = 0;

        for i in 0..values.len() {
            let mut intermed_val = (values[i] * self.crt_inv_vals[i]) % self.primes[i];
            intermed_val = ((intermed_val * self.crt_vals[i]) as u128 % self.modulus) as u64;
            regular_rep_val = ((regular_rep_val + intermed_val) as u128 % self.modulus) as u64;
        }

        regular_rep_val
    }
}

// Helper functions (you might want to put these in a separate module)
fn is_prime(n: u64) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crt_context() {
        let ctx = CRTContext::new(3, 20, 1024);

        // Test CRT and reconstruction
        let value = 12345678;
        let crt_value = ctx.crt(value);
        let reconstructed = ctx.reconstruct(&crt_value);

        assert_eq!(value, reconstructed);
    }

    #[test]
    fn test_prime_generation() {
        let ctx = CRTContext::new(5, 20, 1024);

        assert_eq!(ctx.primes.len(), 5);
        for &prime in &ctx.primes {
            assert!(is_prime(prime));
            assert_eq!(prime % (2 * 1024), 1);
        }
    }
}
