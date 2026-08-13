pub trait Zero {
    const ZERO: Self;
}

impl Zero for f32 {
    const ZERO: Self = 0f32;
}

impl Zero for u32 {
    const ZERO: Self = 0u32;
}

impl Zero for i32 {
    const ZERO: Self = 0i32;
}

pub trait One {
    const ONE: Self;
}

impl One for f32 {
    const ONE: Self = 1f32;
}

impl One for u32 {
    const ONE: Self = 1u32;
}

impl One for i32 {
    const ONE: Self = 1i32;
}
