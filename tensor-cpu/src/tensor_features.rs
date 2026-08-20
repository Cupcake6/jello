use crate::CpuBackend as B;
use tensor_core::{backend::tensor_features::FlatIter, dtype::SupportedDType};

pub struct CpuFlatIter<'a, T: SupportedDType<B>>(std::slice::Iter<'a, T>);

impl<'a, T: SupportedDType<B>> Iterator for CpuFlatIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().copied()
    }
}

impl FlatIter for B {
    fn flat_iter<'a, T: SupportedDType<Self>>(
        tensor: &'a Self::TensorImpl<T>,
    ) -> impl Iterator<Item = T> {
        CpuFlatIter(tensor.data.iter())
    }
}
