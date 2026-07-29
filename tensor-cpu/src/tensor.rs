use crate::Cpu;
use smallvec::SmallVec;
use tensor_core::{backend::TensorOps, dtype::SupportedDType, tensor_metadata::TensorMetadata};

pub struct CpuTensor<T: SupportedDType<Cpu>> {
    data: SmallVec<[T; 1]>,
    metadata: TensorMetadata,
}

impl TensorOps for Cpu {
    type Tensor<T: SupportedDType<Self>> = CpuTensor<T>;
}
