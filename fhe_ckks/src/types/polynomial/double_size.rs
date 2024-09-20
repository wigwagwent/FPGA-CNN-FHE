use num_traits::{FromPrimitive, PrimInt, Signed};

pub trait DoubleSized: Copy + Sized {
    type Double: PrimInt + Signed + FromPrimitive;

    fn to_double(self) -> Self::Double;
    fn from_double(value: Self::Double) -> Self;
}

impl DoubleSized for i8 {
    type Double = i16;

    fn to_double(self) -> Self::Double {
        self as i16
    }

    fn from_double(value: Self::Double) -> Self {
        value as i8
    }
}

impl DoubleSized for i16 {
    type Double = i32;

    fn to_double(self) -> Self::Double {
        self as i32
    }

    fn from_double(value: Self::Double) -> Self {
        value as i16
    }
}

impl DoubleSized for i32 {
    type Double = i64;

    fn to_double(self) -> Self::Double {
        self as i64
    }

    fn from_double(value: Self::Double) -> Self {
        value as i32
    }
}

impl DoubleSized for i64 {
    type Double = i128;

    fn to_double(self) -> Self::Double {
        self as i128
    }

    fn from_double(value: Self::Double) -> Self {
        value as i64
    }
}

// impl<T> DoubleSized for PolynomialCoefficients<T>
// where
//     T: PrimInt + Signed + Debug + Clone + Copy + DoubleSized,
//     PolynomialCoefficients<T::Double>: PrimInt + Signed + FromPrimitive,
// {
//     type Double = PolynomialCoefficients<T::Double>;

//     fn to_double(self) -> Self::Double {
//         PolynomialCoefficients {
//             coeffs: self.coeffs.to_double(),
//             has_data: self.has_data,
//         }
//     }

//     fn from_double(value: Self::Double) -> Self {
//         PolynomialCoefficients {
//             coeffs: T::from_double(value.coeffs),
//             has_data: value.has_data,
//         }
//     }
// }
