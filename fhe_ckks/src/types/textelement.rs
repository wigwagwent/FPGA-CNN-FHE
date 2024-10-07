use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{PrimInt, Signed};

use super::{
    ciphertext::Ciphertext,
    plaintext::{self, Plaintext},
    polynomial::double_size::DoubleSized,
};

#[derive(Clone)]
pub enum TextElement<T, const N: usize>
where
    T: PrimInt + Signed + DoubleSized,
{
    Plaintext(Plaintext<T, N>),
    Ciphertext(Ciphertext<T, N>),
}

impl<T, const N: usize> Add for TextElement<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (TextElement::Plaintext(a), TextElement::Plaintext(b)) => {
                TextElement::Plaintext(a.add(&b))
            }
            (TextElement::Ciphertext(a), TextElement::Ciphertext(b)) => {
                TextElement::Ciphertext(a.add(&b))
            }
            (TextElement::Plaintext(a), TextElement::Ciphertext(b)) => {
                TextElement::Ciphertext(b.add_plain(&a))
            }
            (TextElement::Ciphertext(b), TextElement::Plaintext(a)) => {
                TextElement::Ciphertext(b.add_plain(&a))
            }
        }
    }
}

impl<T, const N: usize> Sub for TextElement<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (TextElement::Plaintext(a), TextElement::Plaintext(b)) => {
                TextElement::Plaintext(a.subtract(&b))
            }
            (TextElement::Ciphertext(a), TextElement::Ciphertext(b)) => {
                TextElement::Ciphertext(a.subtract(&b))
            }
            (TextElement::Plaintext(a), TextElement::Ciphertext(b)) => {
                TextElement::Ciphertext(b.subtract_plain(&a))
            }
            (TextElement::Ciphertext(b), TextElement::Plaintext(a)) => {
                TextElement::Ciphertext(b.subtract_plain(&a))
            }
        }
    }
}

impl<T, const N: usize> AddAssign for TextElement<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<T, const N: usize> SubAssign for TextElement<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl<T, const N: usize> Mul for TextElement<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (TextElement::Plaintext(a), TextElement::Plaintext(b)) => {
                TextElement::Plaintext(a.multiply(&b))
            }
            (TextElement::Ciphertext(a), TextElement::Ciphertext(b)) => {
                TextElement::Ciphertext(a.multiply(&b))
            }
            (TextElement::Plaintext(a), TextElement::Ciphertext(b)) => {
                TextElement::Ciphertext(b.multiply_plain(&a))
            }
            (TextElement::Ciphertext(b), TextElement::Plaintext(a)) => {
                TextElement::Ciphertext(b.multiply_plain(&a))
            }
        }
    }
}

impl<T, const N: usize> MulAssign for TextElement<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<T, const N: usize> TextElement<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    pub fn rotate(&self, amount: usize) -> Self {
        match self {
            TextElement::Plaintext(plaintext) => TextElement::Plaintext(plaintext.rotate(amount)),
            TextElement::Ciphertext(ciphertext) => {
                TextElement::Ciphertext(ciphertext.rotate(amount))
            }
        }
    }

    pub fn encrypt(&self) -> Ciphertext<T, N> {
        match self {
            TextElement::Plaintext(plaintext) => plaintext.encrypt(),
            TextElement::Ciphertext(ciphertext) => ciphertext.clone(),
        }
    }

    pub fn decrypt(&self) -> Plaintext<T, N> {
        match self {
            TextElement::Plaintext(plaintext) => plaintext.clone(),
            TextElement::Ciphertext(ciphertext) => ciphertext.decrypt(),
        }
    }
}

impl<T, const N: usize> From<Plaintext<T, N>> for TextElement<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    fn from(plaintext: Plaintext<T, N>) -> Self {
        TextElement::Plaintext(plaintext)
    }
}

impl<T, const N: usize> From<Ciphertext<T, N>> for TextElement<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    fn from(ciphertext: Ciphertext<T, N>) -> Self {
        TextElement::Ciphertext(ciphertext)
    }
}
