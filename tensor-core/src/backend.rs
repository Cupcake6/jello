use crate::dtype::SupportedDType;

pub trait Backend: TensorOps {}

pub trait TensorOps {
    type Tensor<T: SupportedDType<Self>>;
}
