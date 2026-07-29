use tensor_core::backend::Backend;

pub mod tensor;

pub struct Cpu;

impl Backend for Cpu {}
