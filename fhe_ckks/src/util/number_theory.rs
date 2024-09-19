use rand::Rng;

/// Computes an exponent in a modulus.
fn mod_exp(val: usize, exp: usize, modulus: usize) -> usize {
    let mut result = 1;
    let mut base = val % modulus;
    let mut exponent = exp;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = (result * base) % modulus;
        }
        exponent >>= 1;
        base = (base * base) % modulus;
    }
    result
}

/// Finds an inverse in a given prime modulus.
fn mod_inv(val: usize, modulus: usize) -> usize {
    mod_exp(val, modulus - 2, modulus)
}

/// Finds a generator in the given modulus.
fn find_generator(modulus: usize) -> Option<usize> {
    for g in 2..modulus {
        if is_primitive_root(g, modulus) {
            return Some(g);
        }
    }
    None
}

/// Finds a root of unity in the given modulus.
fn root_of_unity(order: usize, modulus: usize) -> Result<usize, String> {
    if (modulus - 1) % order != 0 {
        return Err(format!("Must have order q | m - 1, where m is the modulus. The values m = {} and q = {} do not satisfy this.", modulus, order));
    }

    let generator = find_generator(modulus).ok_or("No primitive root of unity mod m")?;
    let exp = (modulus - 1) / order;
    let result = mod_exp(generator, exp, modulus);

    if result == 1 {
        root_of_unity(order, modulus)
    } else {
        Ok(result)
    }
}

/// Determines whether a number is prime.
fn is_prime(number: usize, num_trials: u32) -> bool {
    if number < 2 {
        return false;
    }
    if number != 2 && number % 2 == 0 {
        return false;
    }

    let mut exp = number - 1;
    while exp % 2 == 0 {
        exp /= 2;
    }

    let mut rng = rand::thread_rng();

    for _ in 0..num_trials {
        let rand_val = rng.gen_range(1..number);
        let mut new_exp = exp;
        let mut power = mod_exp(rand_val, new_exp, number);

        while new_exp != number - 1 && power != 1 && power != number - 1 {
            power = (power * power) % number;
            new_exp *= 2;
        }

        if power != number - 1 && new_exp % 2 == 0 {
            return false;
        }
    }

    true
}

/// Helper function to check if a number is a primitive root modulo p
fn is_primitive_root(a: usize, p: usize) -> bool {
    let factors = prime_factors(p - 1);
    for factor in factors {
        if mod_exp(a, (p - 1) / factor, p) == 1 {
            return false;
        }
    }
    true
}

/// Helper function to find prime factors of a number
fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut i = 2;

    while i * i <= n {
        if n % i == 0 {
            factors.push(i);
            n /= i;
        } else {
            i += 1;
        }
    }

    if n > 1 {
        factors.push(n);
    }

    factors
}
