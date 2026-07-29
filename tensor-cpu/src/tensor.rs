use crate::Cpu;
use smallvec::{SmallVec, smallvec};
use tensor_core::{
    backend::TensorOps,
    dtype::SupportedDType,
    tensor_metadata::{TensorMetadata, shape::Shape},
};

pub struct CpuTensor<T: SupportedDType<Cpu>> {
    data: SmallVec<[T; 1]>,
    metadata: TensorMetadata,
}

impl TensorOps for Cpu {
    type Tensor<T: SupportedDType<Self>> = CpuTensor<T>;

    fn full<T: SupportedDType<Self>>(fill_value: T, shape: Shape) -> Self::Tensor<T> {
        let metadata = TensorMetadata::new(shape);
        let data = smallvec![fill_value; metadata.num_items() as usize];

        Self::Tensor { data, metadata }
    }

    fn num_dims<T: SupportedDType<Self>>(tensor: &Self::Tensor<T>) -> usize {
        tensor.metadata.num_dims()
    }
}
