pub use tensor_cpu::Cpu;

use tensor_core::backend::{BackendKind, TensorOps};

pub trait Backend: BackendKind + TensorOps {}
impl<T: BackendKind + TensorOps> Backend for T {}
