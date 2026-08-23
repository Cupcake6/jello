use crate::{backend::Backend, dtype::SupportedDType};

pub mod error {}

pub trait FlatIter: Backend {
    fn flat_iter<T: SupportedDType<Self>>(tensor: &Self::TensorImpl<T>) -> impl Iterator<Item = T>;
}
