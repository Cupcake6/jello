use tensor::prelude::*;

fn contiguous_stride<const N: usize>(shape: [u64; N]) -> [u64; N] {
    let mut output = [0; N];
    let mut suffix_product = 1;

    for i in (0..shape.len()).rev() {
        output[i] = suffix_product;
        suffix_product *= shape[i];
    }

    output
}

fn assert_metadata<B, T, const N: usize>(fill_value: T, shape: [u64; N])
where
    B: Backend,
    T: SupportedDType<B>,
{
    let tensor = Tensor::full(fill_value, shape);
    assert_eq!(tensor.num_dims(), shape.len());
    assert_eq!(tensor.num_items(), shape.iter().product());
    assert_eq!(***tensor.shape(), shape);
    assert_eq!(***tensor.stride(), contiguous_stride(shape))
}

#[test]
fn test_0d() {
    assert_metadata::<Cpu, _, _>(3.1f32, []);
    assert_metadata::<Cpu, _, _>(4u32, []);
    assert_metadata::<Cpu, _, _>(-8i32, []);
    assert_metadata::<Cpu, _, _>(false, []);
}

#[test]
fn test_1d() {
    assert_metadata::<Cpu, _, _>(3.1f32, [2]);
    assert_metadata::<Cpu, _, _>(4u32, [0]);
    assert_metadata::<Cpu, _, _>(-8i32, [4]);
    assert_metadata::<Cpu, _, _>(false, [1]);
}

#[test]
fn test_2d() {
    assert_metadata::<Cpu, _, _>(3.1f32, [4, 2]);
    assert_metadata::<Cpu, _, _>(4u32, [0, 1]);
    assert_metadata::<Cpu, _, _>(-8i32, [0, 0]);
    assert_metadata::<Cpu, _, _>(false, [3, 0]);
}

#[test]
fn test_3d() {
    assert_metadata::<Cpu, _, _>(3.1f32, [2, 2, 3]);
    assert_metadata::<Cpu, _, _>(4u32, [0, 2, 8]);
    assert_metadata::<Cpu, _, _>(-8i32, [0, 0, 0]);
    assert_metadata::<Cpu, _, _>(false, [1, 1, 1]);
}

#[test]
fn test_4d() {
    assert_metadata::<Cpu, _, _>(3.1f32, [0, 0, 0, 0]);
    assert_metadata::<Cpu, _, _>(4u32, [2, 3, 2, 4]);
    assert_metadata::<Cpu, _, _>(-8i32, [1, 1, 1, 1]);
    assert_metadata::<Cpu, _, _>(false, [4, 0, 8, 2]);
}
