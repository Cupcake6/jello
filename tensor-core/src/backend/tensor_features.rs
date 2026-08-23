use crate::{backend::Backend, dtype::SupportedDType};

pub mod error {
    #[derive(Debug, thiserror::Error)]
    pub enum IndexingError {
        #[error("mismatched number of dimensions: expected {0}, but {1} were provided")]
        MismatchedNumberOfDimensions(usize, usize),

        #[error("index out of bounds")]
        IndexOutOfBounds,
    }
}

pub trait FlatIter: Backend {
    fn flat_iter<T: SupportedDType<Self>>(tensor: &Self::TensorImpl<T>) -> impl Iterator<Item = T>;
}

pub trait Indexing: Backend {
    fn get<T: SupportedDType<Self>>(
        tensor: &Self::TensorImpl<T>,
    ) -> Result<T, error::IndexingError>;
}
