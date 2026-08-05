use crate::CpuBackend as B;
use tensor_core::{backend::tensor_features::FlatIter, dtype::SupportedDType};

pub struct CpuFlatIter<'a, T: SupportedDType<B>>(std::slice::Iter<'a, T>);

impl<'a, T: SupportedDType<B>> Iterator for CpuFlatIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl FlatIter for B {
    type Iterator<'a, T: SupportedDType<Self>> = CpuFlatIter<'a, T>;

    fn flat_iter<'a, T: SupportedDType<Self>>(
        tensor: &'a Self::TensorPrimitive<T>,
    ) -> Self::Iterator<'a, T> {
        CpuFlatIter(tensor.data.iter())
    }
}
