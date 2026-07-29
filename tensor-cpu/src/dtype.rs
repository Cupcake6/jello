use crate::Cpu;
use tensor_core::dtype::SupportedDType;

impl SupportedDType<Cpu> for f32 {}
impl SupportedDType<Cpu> for u32 {}
impl SupportedDType<Cpu> for i32 {}
impl SupportedDType<Cpu> for bool {}
