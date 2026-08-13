use crate::CpuBackend as B;
use tensor_core::{backend::tensor_features::FlatIter, dtype::SupportedDType};

pub struct CpuFlatIter<'a, T: SupportedDType<B>>(std::slice::Iter<'a, T>);

impl<'a, T: SupportedDType<B>> Iterator for CpuFlatIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct CpuFlatIterMut<'a, T: SupportedDType<B>>(std::slice::IterMut<'a, T>);

impl<'a, T: SupportedDType<B>> Iterator for CpuFlatIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl FlatIter for B {
    fn flat_iter<'a, T: SupportedDType<Self>>(
        tensor: &'a Self::TensorImpl<T>,
    ) -> impl Iterator<Item = &'a T> {
        CpuFlatIter(tensor.data.iter())
    }

    fn flat_iter_mut<'a, T: SupportedDType<Self>>(
        tensor: &'a mut Self::TensorImpl<T>,
    ) -> impl Iterator<Item = &'a mut T> {
        CpuFlatIterMut(tensor.data.iter_mut())
    }
}
