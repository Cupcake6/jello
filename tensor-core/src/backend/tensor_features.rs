use crate::{backend::Backend, dtype::SupportedDType};

pub trait FlatIter: Backend {
    type Iterator<'a>: Iterator;
    fn flat_iter<'a, T: SupportedDType<Self>>(
        tensor: &'a Self::TensorPrimitive<T>,
    ) -> Self::Iterator<'a>;
}
