use crate::{Ciphertext, Plaintext};
use std::ops::Add;

mod add;

// Implement Add for Ciphertext + Ciphertext
impl Add for Ciphertext {
    type Output = Ciphertext;

    fn add(self, other: Self) -> Self::Output {
        add::add_ciphertext_ciphertext(self, other)
    }
}

// Implement Add for &Ciphertext + &Ciphertext
impl Add for &Ciphertext {
    type Output = Ciphertext;

    fn add(self, other: Self) -> Self::Output {
        add::add_ciphertext_ciphertext_ref(&self, &other)
    }
}

// Implement Add for Ciphertext + Plaintext
impl Add<Plaintext> for Ciphertext {
    type Output = Ciphertext;

    fn add(self, other: Plaintext) -> Self::Output {
        add::add_ciphertext_plaintext(self, other)
    }
}

// Implement Add for &Ciphertext + &Plaintext
impl Add<&Plaintext> for &Ciphertext {
    type Output = Ciphertext;

    fn add(self, other: &Plaintext) -> Self::Output {
        add::add_ciphertext_plaintext_ref(&self, &other)
    }
}

// Implement Add for Plaintext + Ciphertext
impl Add<Ciphertext> for Plaintext {
    type Output = Ciphertext;

    fn add(self, other: Ciphertext) -> Self::Output {
        add::add_ciphertext_plaintext(other, self)
    }
}

// Implement Add for &Plaintext + &Ciphertext
impl Add<&Ciphertext> for &Plaintext {
    type Output = Ciphertext;

    fn add(self, other: &Ciphertext) -> Self::Output {
        add::add_ciphertext_plaintext_ref(&other, &self)
    }
}

// Implement Add for Plaintext + Plaintext
impl Add for Plaintext {
    type Output = Plaintext;

    fn add(self, other: Self) -> Self::Output {
        add::add_plaintext_plaintext(&self, &other)
    }
}

// Implement Add for &Plaintext + &Plaintext
impl Add for &Plaintext {
    type Output = Plaintext;

    fn add(self, other: Self) -> Self::Output {
        add::add_plaintext_plaintext(&self, &other)
    }
}
