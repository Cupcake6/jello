use tensor_core::backend::BackendKind;

mod dtype;
mod tensor;

pub struct Cpu;

impl BackendKind for Cpu {}
