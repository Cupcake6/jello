use crate::{backend::Backend, dtype::SupportedDType};

pub trait FlatIter: Backend {
    type Iterator<'a, T: SupportedDType<Self>>: Iterator<Item = &'a T>;

    fn flat_iter<'a, T: SupportedDType<Self>>(
        tensor: &'a Self::TensorPrimitive<T>,
    ) -> Self::Iterator<'a, T>;
}
