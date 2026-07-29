use tensor_core::{
    backend::{Backend, TensorOps},
    dtype::SupportedDType,
};

type RawTensor<B, T> = <B as TensorOps>::Tensor<T>;

pub struct Tensor<B: Backend, T: SupportedDType<B>>(RawTensor<B, T>);

impl<B: Backend, T: SupportedDType<B>> Tensor<B, T> {}
