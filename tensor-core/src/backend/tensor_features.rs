use crate::{backend::Backend, dtype::SupportedDType};

pub trait FlatIter: Backend {
    fn flat_iter<'a, T: SupportedDType<Self>>(
        tensor: &'a Self::TensorPrimitive<T>,
    ) -> impl Iterator<Item = &'a T>;
}
