use tensor_core::backend::Backend;

mod dtype;
mod tensor;

pub struct Cpu;

impl Backend for Cpu {}
