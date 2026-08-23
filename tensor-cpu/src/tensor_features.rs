use crate::CpuBackend as B;
use tensor_core::{backend::tensor_features::FlatIter, dtype::SupportedDType};

pub enum CpuFlatIter<'a, T: SupportedDType<B>> {
    Contiguous(std::slice::Iter<'a, T>),
    General,
}

impl<'a, T: SupportedDType<B>> Iterator for CpuFlatIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(it) => it.next().copied(),
            Self::General => todo!(),
        }
    }
}

impl FlatIter for B {
    fn flat_iter<'a, T: SupportedDType<Self>>(
        tensor: &'a Self::TensorImpl<T>,
    ) -> impl Iterator<Item = T> {
        if tensor.metadata.contiguous() {
            CpuFlatIter::Contiguous(tensor.data.iter())
        } else {
            CpuFlatIter::General
        }
    }
}
