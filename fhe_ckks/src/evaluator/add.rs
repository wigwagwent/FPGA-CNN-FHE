use crate::{Ciphertext, Plaintext};

pub fn add_ciphertext_ciphertext(a: Ciphertext, b: Ciphertext) -> Ciphertext {
    assert!(
        a.scaling_factor == b.scaling_factor, "{}", format!(
            "Scaling factors are not equal. Ciphertext 1 scaling factor: {} bits, Ciphertext 2 scaling factor: {} bits",
            (a.scaling_factor as f64).log2(),
            (b.scaling_factor as f64).log2()
        ));

    assert!(
        a.modulus == b.modulus,
        "{}",
        format!(
            "Moduli are not equal. Ciphertext 1 modulus: {} bits, Ciphertext 2 modulus: {} bits",
            (a.modulus.bits() as f64).log2(),
            (b.modulus.bits() as f64).log2()
        )
    );

    let c0 = (a.c0 + b.c0) % &a.modulus;
    let c1 = (a.c1 + b.c1) % &a.modulus;

    Ciphertext::new(c0, c1, a.scaling_factor, a.modulus)
}

pub fn add_ciphertext_ciphertext_ref(a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
    assert!(
        a.scaling_factor == b.scaling_factor, "{}", format!(
            "Scaling factors are not equal. Ciphertext 1 scaling factor: {} bits, Ciphertext 2 scaling factor: {} bits",
            (a.scaling_factor as f64).log2(),
            (b.scaling_factor as f64).log2()
        ));

    assert!(
        a.modulus == b.modulus,
        "{}",
        format!(
            "Moduli are not equal. Ciphertext 1 modulus: {} bits, Ciphertext 2 modulus: {} bits",
            (a.modulus.bits() as f64).log2(),
            (b.modulus.bits() as f64).log2()
        )
    );

    let c0 = (&a.c0 + &b.c0) % &a.modulus;
    let c1 = (&a.c1 + &b.c1) % &a.modulus;

    Ciphertext::new(c0, c1, a.scaling_factor, a.modulus.clone())
}

pub fn add_ciphertext_plaintext(a: Ciphertext, b: Plaintext) -> Ciphertext {
    assert!(
        a.scaling_factor == b.scaling_factor, "{}", format!(
            "Scaling factors are not equal. Ciphertext 1 scaling factor: {} bits, Ciphertext 2 scaling factor: {} bits",
            (a.scaling_factor as f64).log2(),
            (b.scaling_factor as f64).log2()
        ));

    let c0 = (a.c0 + b.poly) % &a.modulus;

    Ciphertext::new(c0, a.c1, a.scaling_factor, a.modulus)
}

pub fn add_ciphertext_plaintext_ref(a: &Ciphertext, b: &Plaintext) -> Ciphertext {
    assert!(
        a.scaling_factor == b.scaling_factor, "{}", format!(
            "Scaling factors are not equal. Ciphertext 1 scaling factor: {} bits, Ciphertext 2 scaling factor: {} bits",
            (a.scaling_factor as f64).log2(),
            (b.scaling_factor as f64).log2()
        ));

    let c0 = (&a.c0 + &b.poly) % &a.modulus;

    Ciphertext::new(c0, a.c1.clone(), a.scaling_factor, a.modulus.clone())
}

pub fn add_plaintext_plaintext(a: &Plaintext, b: &Plaintext) -> Plaintext {
    assert!(
        a.scaling_factor == b.scaling_factor, "{}", format!(
            "Scaling factors are not equal. Plaintext 1 scaling factor: {} bits, Plaintext 2 scaling factor: {} bits",
            (a.scaling_factor as f64).log2(),
            (b.scaling_factor as f64).log2()
        ));

    Plaintext::new(&a.poly + &b.poly, a.scaling_factor)
}
