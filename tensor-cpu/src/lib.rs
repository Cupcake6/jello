use tensor_core::backend::Backend;

mod tensor;

pub struct Cpu;

impl Backend for Cpu {}
