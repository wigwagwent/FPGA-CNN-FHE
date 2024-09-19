//mod evaluator;
//mod util;
mod types;

pub use types::ciphertext::Ciphertext;
pub use types::plaintext::Plaintext;
pub use types::polynomial::Polynomial;

#[cfg(feature = "i8")]
pub type CoeffType = i8;

#[cfg(feature = "i16")]
pub type CoeffType = i16;

#[cfg(feature = "i32")]
pub type CoeffType = i32;

#[cfg(feature = "i64")]
pub type CoeffType = i64;
