use rstest::rstest;
use tensor::prelude::*;
use tensor_cpu::CpuBackend;

type B = CpuBackend;

fn contiguous_stride<const N: usize>(shape: [u64; N]) -> [u64; N] {
    let mut output = [0; N];
    let mut suffix_product = 1;

    for i in (0..shape.len()).rev() {
        output[i] = suffix_product;
        suffix_product *= shape[i];
    }

    output
}

#[rstest]
fn assert_metadata<T, const N: usize>(
    #[values(42.42f32, 123u32, -23i32, false, true)] fill_value: T,
    #[values(
        [],
        [0],
        [1],
        [4],
        [0, 0],
        [3, 0],
        [1, 1],
        [1, 8],
        [2, 3],
        [0, 0, 0],
        [0, 4, 5],
        [1, 1, 1],
        [8, 9, 1],
        [2, 9, 4],
        [0, 0, 0, 0],
        [5, 4, 0, 3],
        [1, 1, 1, 1],
        [8, 1, 9, 5],
        [4, 8, 3, 9],
    )]
    shape: [u64; N],
) where
    T: SupportedDType<B>,
{
    let tensor = Tensor::full(fill_value, shape);
    assert_eq!(tensor.num_dims(), shape.len());
    assert_eq!(tensor.num_items(), shape.iter().product());
    assert_eq!(***tensor.shape(), shape);
    assert_eq!(***tensor.stride(), contiguous_stride(shape));
    assert_eq!(tensor.dtype(), <T as SupportedDType<B>>::DTYPE);
}
