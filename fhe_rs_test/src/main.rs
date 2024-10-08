use fhe::bfv::{
    BfvParameters, BfvParametersBuilder, Ciphertext, Encoding, Plaintext, PublicKey, SecretKey,
};
use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};
use rand::thread_rng;
use std::sync::Arc;

const SCALE_FACTOR: i64 = 1 << 14; // Use 14 bits for fractional part
const MAX_VALUE: i64 = (1 << 15) - 1; // Maximum value for 16-bit signed integer
const MIN_VALUE: i64 = -(1 << 15); // Minimum value for 16-bit signed integer

fn encrypt_fixed_point(
    value: f64,
    public_key: &PublicKey,
    parameters: &Arc<BfvParameters>,
) -> Ciphertext {
    let fixed_point = convert_to_fixed_point(value);
    let plaintext = Plaintext::try_encode(&[fixed_point], Encoding::poly(), parameters).unwrap();
    public_key
        .try_encrypt(&plaintext, &mut thread_rng())
        .unwrap()
}

fn convert_to_fixed_point(value: f64) -> i64 {
    let scaled_value = (value * SCALE_FACTOR as f64).round() as i64;
    assert!(scaled_value >= MIN_VALUE && scaled_value <= MAX_VALUE);
    scaled_value
}

// fn decrypt_fixed_point(ciphertext: &Ciphertext, secret_key: &SecretKey) -> f64 {
//     let decrypted_plaintext = secret_key.try_decrypt(ciphertext).unwrap();
//     let decrypted_vec = Vec::<i64>::try_decode(&decrypted_plaintext, Encoding::poly()).unwrap();

//     // let clamped_result = (decrypted_vec[0] as i64).clamp(MIN_VALUE, MAX_VALUE);
//     // let result = clamped_result as f64 / SCALE_FACTOR as f64;
//     // if clamped_result != decrypted_vec[0] as i64 {
//     //     // println!("Value was clamped to: {}", result);
//     // }
//     // result
//     decrypted_vec[0]
// }

fn decrypt_i64(ciphertext: &Ciphertext, secret_key: &SecretKey) -> i64 {
    let decrypted_plaintext = secret_key.try_decrypt(ciphertext).unwrap();
    let decrypted_vec = Vec::<i64>::try_decode(&decrypted_plaintext, Encoding::poly()).unwrap();

    // let clamped_result = (decrypted_vec[0] as i64).clamp(MIN_VALUE, MAX_VALUE);
    // let result = clamped_result as f64 / SCALE_FACTOR as f64;
    // if clamped_result != decrypted_vec[0] as i64 {
    //     // println!("Value was clamped to: {}", result);
    // }
    // result
    decrypted_vec[0]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parameters = BfvParametersBuilder::new()
        .set_degree(4096)
        .set_moduli(&[
            0x7fffffffba0001, // 55 bits
            0x7fffffffaa0001, // 55 bits
            0x7fffffff7e0001, // 55 bits
            0x7fffffff380001, // 55 bits (added an extra modulus)
        ])
        .set_plaintext_modulus(1 << 32) // 32-bit plaintext modulus
        .build_arc()?;

    let secret_key = SecretKey::random(&parameters, &mut thread_rng());
    let public_key = PublicKey::new(&secret_key, &mut thread_rng());

    let a = 0.543623;
    let b = 0.34543;

    let encrypted_a = encrypt_fixed_point(a, &public_key, &parameters);
    let encrypted_b = encrypt_fixed_point(b, &public_key, &parameters);

    let mut encrypted_product = &encrypted_a * &encrypted_b;
    let decrypted_product = decrypt_i64(&encrypted_product, &secret_key);

    println!("Original numbers: a = {}, b = {}", a, b);
    println!("Encrypted multiplication result: {}", decrypted_product);

    for i in 1..=3 {
        if let Ok(()) = encrypted_product.mod_switch_to_next_level() {
            let decrypted_scaled = decrypt_i64(&encrypted_product, &secret_key);
            println!("Scaled result (level {}): {}", i, decrypted_scaled);
        } else {
            println!("Cannot perform further modulus switching");
            break;
        }
    }

    println!(
        "Actual multiplication result (Fixed Raw): {}",
        convert_to_fixed_point(a) * convert_to_fixed_point(b)
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    const SCALE_FACTOR: i64 = 16384; // 2^14
    const MAX_VALUE: i32 = 32767; // 2^15 - 1
    const MIN_VALUE: i32 = -32768; // -2^15
    const SCALE_RECIPROCAL: f64 = 1.0 / SCALE_FACTOR as f64;

    fn fixed_point_multiply(a: f64, b: f64) -> f64 {
        // Convert to fixed-point with higher precision (32-bit instead of 16-bit)
        let fixed_a = (a * SCALE_FACTOR as f64).round() as i64;
        let fixed_b = (b * SCALE_FACTOR as f64).round() as i64;
        println!("Fixed-point values: a = {}, b = {}", fixed_a, fixed_b);

        // Perform multiplication
        let product = multiply_without_overflow(fixed_a, fixed_b);
        println!("Raw product: {}", product);

        // Scale down
        let scaled = scale_down(product);
        println!("Scaled product: {}", scaled);

        // Clamp the result to 16-bit range
        let clamped_product = if scaled > MAX_VALUE as i64 {
            MAX_VALUE
        } else if scaled < MIN_VALUE as i64 {
            MIN_VALUE
        } else {
            scaled as i32
        };
        println!("Clamped product: {}", clamped_product);

        // Convert back to floating-point
        clamped_product as f64 * SCALE_RECIPROCAL
    }

    fn multiply_without_overflow(a: i64, b: i64) -> i64 {
        let mut result = 0;
        let mut a = a.abs();
        let mut b = b.abs();

        while a > 0 {
            if a % 2 != 0 {
                result = add_without_overflow(result, b);
            }
            b = add_without_overflow(b, b);
            a = a / 2;
        }

        if (a < 0) != (b < 0) {
            -result
        } else {
            result
        }
    }

    fn add_without_overflow(a: i64, b: i64) -> i64 {
        let mut result = 0;
        let mut carry = 0;
        let mut place_value = 1;

        while place_value <= a || place_value <= b || carry != 0 {
            let bit_a = (a / place_value) % 2;
            let bit_b = (b / place_value) % 2;
            let sum = bit_a + bit_b + carry;
            result += (sum % 2) * place_value;
            carry = sum / 2;
            place_value = place_value * 2;
        }

        result
    }

    fn scale_down(value: i64) -> i64 {
        value / SCALE_FACTOR
    }

    #[test]
    fn test_fixed_point_multiplication() {
        let a = 0.543623;
        let b = 0.34543;

        let result = fixed_point_multiply(a, b);
        let expected = a * b;

        println!("Fixed-point multiplication result: {}", result);
        println!("Actual multiplication result: {}", expected);

        // Check if the result is close to the expected value
        assert!(
            (result - expected).abs() < 0.0001,
            "Results are too far apart"
        );
    }
}
