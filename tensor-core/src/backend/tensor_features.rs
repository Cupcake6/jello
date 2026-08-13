use crate::{backend::Backend, dtype::SupportedDType};

pub trait FlatIter: Backend {
    fn flat_iter<'a, T: SupportedDType<Self>>(
        tensor: &'a Self::TensorImpl<T>,
    ) -> impl Iterator<Item = &'a T>;

    fn flat_iter_mut<'a, T: SupportedDType<Self>>(
        tensor: &'a mut Self::TensorImpl<T>,
    ) -> impl Iterator<Item = &'a mut T>;
}
