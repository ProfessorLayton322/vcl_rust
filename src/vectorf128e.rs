//This file compiles everywhere

#[cfg(
    not(all(
        any(
            target_arch = "x86",
            target_arch = "x86_64"
        ),
        target_feature="sse2"
    ))
)]
pub mod vec128e {

pub struct Vec4f {
    data: [f32; 4],
}

impl Vec4f {
    pub fn from_scalar(value: f32) -> Self {
        Self {
            data: [value; 4]
        }
    }

    pub fn size(&self) -> usize {
        4
    }
}

}

#[cfg(
    all(
        any(
            target_arch = "x86",
            target_arch = "x86_64"
        ),
        target_feature="sse2"
    )
)]
pub mod vec128e {
}
