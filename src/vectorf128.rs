//This file should only be included if SSE feature is supported and architecture is x86(_64)

#[cfg(
    all(
        any(
            target_arch = "x86",
            target_arch = "x86_64"
        ),
        target_feature = "sse2"
    )
)]
pub mod vec128 {

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use aligned_array::{Aligned, A16};

    pub fn selectf(s: __m128, a: __m128, b: __m128) -> __m128 {
        #[cfg(target_feature = "sse4.1")] {
            unsafe { _mm_blendv_ps(b, a, s) }
        }
        #[cfg(not(target_feature = "sse4.1"))] {
            unsafe {
                _mm_or_ps(
                    _mm_and_ps(s, a),
                    _mm_andnot_ps(s, b)
                )
            }
        }
    }

    #[derive(Clone, Debug, Copy)]
    pub struct Vec4f {
        xmm: __m128,
    }

    impl Vec4f {
        pub fn from_scalar(value: f32) -> Self {
            Self {
                xmm: unsafe { _mm_set1_ps(value) }
            }
        }

        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            Self {
                xmm: unsafe { _mm_setr_ps(a, b, c, d) }
            }
        }

        pub fn store(self, p: *mut f32) {
            unsafe { _mm_storeu_ps(p, self.xmm) }
        }

        pub fn store_a(self, p: *mut f32) {
            unsafe { _mm_store_ps(p, self.xmm) }
        }

        pub fn store_nt(self, p: *mut f32) {
            unsafe { _mm_stream_ps(p, self.xmm) }
        }

        pub fn store_partial(self, size: usize, p: *mut f32) {
            let mut aligned_buffer : Aligned<A16, _> = Aligned([0.0f32; 4]);
            self.store_a(aligned_buffer.as_mut_ptr());
            for i in 0..std::cmp::min(4, size) {
                unsafe { *(p.add(i)) = aligned_buffer[i] }
            }
        }

        pub fn load(&mut self, p: *const f32) {
            self.xmm = unsafe { _mm_loadu_ps(p) };
        }

        pub fn load_a(&mut self, p: *const f32) {
            self.xmm = unsafe { _mm_load_ps(p) };
        }

        pub fn load_partial(&mut self, size: usize, p: *const f32) {
            match size {
                0 => *self = Self::default(),
                1 => self.xmm = unsafe { _mm_load_ss(p) },
                2 => self.xmm = unsafe { _mm_setr_ps(*p, *(p.add(1)), 0.0, 0.0) },
                3 => self.xmm = unsafe { _mm_setr_ps(*p, *(p.add(1)), *(p.add(2)), 0.0) },
                4 => self.load(p),
                _ => {},
            };
        }

        pub fn insert(&mut self, index: usize, value: f32) {
            #[cfg(target_feature = "sse4.1")]
            {
                self.xmm = unsafe { _mm_insert_ps(self.xmm, _mm_set_ss(value), ((index & 3) as i32) << 4) };
            }
            #[cfg(not(target_feature = "sse4.1"))]
            {
                let maskl : [i32; 8] = [0, 0, 0, 0, -1, 0, 0, 0];
                let float_mask : *const f32 = unsafe { maskl.as_ptr().add(4 - (index & 3)).cast() };
                let broad : __m128 = unsafe { _mm_set1_ps(value ) };
                let mask : __m128 = unsafe { _mm_loadu_ps(float_mask) };
                self.xmm = selectf(mask, broad, self.xmm);
            }
        }

        pub fn extract(&self, index: usize) -> f32 {
            let mut aligned_buffer : Aligned<A16, _> = Aligned([0.0f32; 4]);
            self.store_a(aligned_buffer.as_mut_ptr());
            aligned_buffer[index & 3]
        }

        pub fn cutoff(&mut self, size: usize) {
            if size >= 4 {
                return;
            }
            let maskl : [i32; 8] = [-1, -1, -1, -1, 0, 0, 0, 0];
            let float_mask : *const f32 = unsafe { maskl.as_ptr().add(4 - size).cast() };
            let mask : __m128 = unsafe { _mm_loadu_ps(float_mask) };
            self.xmm = unsafe { _mm_and_ps(self.xmm, mask) };
        }

        pub const fn size() -> usize {
            4
        }
    }

    impl std::default::Default for Vec4f {
        fn default() -> Self {
            Self::from_scalar(0.0)
        }
    }

    //Indexing would not work because we can't return a reference to xmm register
    /*
    impl std::ops::Index<usize> for Vec4f {
    }
    */

    impl std::ops::Add for Vec4f {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            Self {
                xmm: unsafe { _mm_add_ps(self.xmm, other.xmm) }
            }
        }
    }

    impl std::ops::AddAssign for Vec4f {
        fn add_assign(&mut self, other: Self) {
            self.xmm = unsafe { _mm_add_ps(self.xmm, other.xmm) }
        }
    }

    impl std::ops::Sub for Vec4f {
        type Output = Self;

        fn sub(self, other: Self) -> Self {
            Self {
                xmm: unsafe { _mm_sub_ps(self.xmm, other.xmm) }
            }
        }
    }

    impl std::ops::SubAssign for Vec4f {
        fn sub_assign(&mut self, other: Self) {
           self.xmm = unsafe { _mm_sub_ps(self.xmm, other.xmm) }
        }
    }

    impl std::ops::Neg for Vec4f {
        type Output = Self;

        fn neg(self) -> Self {
            Self {
                xmm : unsafe {_mm_xor_ps(self.xmm, _mm_castsi128_ps(_mm_set1_epi32(i32::MIN)))}
            }
        }
    }

    impl std::ops::Mul for Vec4f {
        type Output = Self;

        fn mul(self, other: Self) -> Self {
            Self {
                xmm: unsafe { _mm_mul_ps(self.xmm, other.xmm) }
            }
        }
    }

    impl std::ops::MulAssign for Vec4f {
        fn mul_assign(&mut self, other: Self) {
            self.xmm = unsafe { _mm_mul_ps(self.xmm, other.xmm) }
        }
    }

    impl std::ops::Div for Vec4f {
        type Output = Self;

        fn div(self, other: Self) -> Self {
            Self {
                xmm: unsafe { _mm_div_ps(self.xmm, other.xmm) }
            }
        }
    }

    impl std::ops::DivAssign for Vec4f {
        fn div_assign(&mut self, other: Self) {
            self.xmm = unsafe { _mm_div_ps(self.xmm, other.xmm) }
        }
    }

    impl std::ops::BitAnd for Vec4f {
        type Output = Self;

        fn bitand(self, other: Self) -> Self {
            Self {
                xmm: unsafe { _mm_and_ps(self.xmm, other.xmm) }
            }
        }
    }

    impl std::ops::BitAndAssign for Vec4f {
        fn bitand_assign(&mut self, other: Self) {
            self.xmm = unsafe { _mm_and_ps(self.xmm, other.xmm) }
        }
    }

    impl std::ops::BitOr for Vec4f {
        type Output = Self;

        fn bitor(self, other: Self) -> Self {
            Self {
                xmm: unsafe { _mm_or_ps(self.xmm, other.xmm) }
            }
        }
    }

    impl std::ops::BitOrAssign for Vec4f {
        fn bitor_assign(&mut self, other: Self) {
            self.xmm = unsafe { _mm_or_ps(self.xmm, other.xmm) }
        }
    }

    impl std::ops::BitXor for Vec4f {
        type Output = Self;

        fn bitxor(self, other: Self) -> Self {
            Self {
                xmm: unsafe { _mm_xor_ps(self.xmm, other.xmm) }
            }
        }
    }

    impl std::ops::BitXorAssign for Vec4f {
        fn bitxor_assign(&mut self, other: Self) {
            self.xmm = unsafe { _mm_xor_ps(self.xmm, other.xmm) }
        }
    }

    impl std::cmp::PartialEq<[f32; 4]> for Vec4f {
        fn eq(&self, other: &[f32; 4]) -> bool {
            let mut arr = [0.0f32; 4];
            self.store(arr.as_mut_ptr());
            arr == *other
        }
    }

    impl std::convert::From<__m128> for Vec4f {
        fn from(value: __m128) -> Self {
            Self {
                xmm : value
            }
        }
    }

    impl std::convert::Into<__m128> for Vec4f {
        fn into(self) -> __m128 {
            self.xmm
        }
    }

    pub fn horizontal_add(vec: Vec4f) -> f32 {
        #[cfg(target_feature = "sse3")] {
            unsafe {
                let t1: __m128 = _mm_hadd_ps(vec.xmm, vec.xmm);
                let t2: __m128 = _mm_hadd_ps(t1, t1);
                _mm_cvtss_f32(t2)
            }
        }
        #[cfg(not(target_feature = "sse3"))] {
            unsafe {
                let t1: __m128 = _mm_movehl_ps(vec.xmm, vec.xmm);
                let t2: __m128 = _mm_add_ps(vec.xmm, t1);
                let t3: __m128 = _mm_shuffle_ps(t2, t2, 1);
                let t4: __m128 = _mm_add_ss(t2, t3);
                _mm_cvtss_f32(t4)
            }
        }
    }

    pub fn max(first: Vec4f, second: Vec4f) -> Vec4f {
        Vec4f {
            xmm: unsafe { _mm_max_ps(first.xmm, second.xmm) }
        }
    }

    pub fn min(first: Vec4f, second: Vec4f) -> Vec4f {
        Vec4f {
            xmm: unsafe { _mm_min_ps(first.xmm, second.xmm) }
        }
    }

    pub fn abs(vec: Vec4f) -> Vec4f {
        let mask : __m128 = unsafe { _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)) };
        Vec4f {
            xmm: unsafe { _mm_and_ps(vec.xmm, mask) }
        }
    }

    const fn mask_helper(i: bool) -> i32 {
        match i {
            true => i32::MIN,
            false => 0,
        }
    }

    pub fn change_sign<const i0: bool, const i1: bool, const i2: bool, const i3: bool>(vec: Vec4f) -> Vec4f {
        if !(i0 | i1 | i2 | i3) {
            return vec;
        }
        let mask: __m128i = unsafe { _mm_setr_epi32(mask_helper(i0), mask_helper(i1), mask_helper(i2), mask_helper(i3)) };
        Vec4f {
            xmm : unsafe { _mm_xor_ps(vec.xmm, _mm_castsi128_ps(mask)) }
        }
    }

    pub fn sign_combine(a : Vec4f, b : Vec4f) -> Vec4f {
        a ^ (b & Vec4f::from_scalar(-0.0f32))
    }

    pub fn sqrt(vec: Vec4f) -> Vec4f {
        Vec4f {
            xmm : unsafe { _mm_sqrt_ps(vec.xmm) }
        }
    }

    pub fn square(vec: Vec4f) -> Vec4f {
        vec * vec
    }

    fn nan_vec() -> Vec4f {
        //These are magic numbers from original Agner Fog's lib
        //https://github.com/vectorclass/version2/blob/master/instrset.h#L415
        Vec4f::from_scalar(f32::from_bits(0x7FC00000 | (0x100 & 0x003FFFFF)))
    }

    pub fn pow(mut vec: Vec4f, mut n: i32) -> Vec4f {
        let return_item = Vec4f::from_scalar(1.0);
        if n < 0 {
            if n == i32::MIN {
                return nan_vec();
            }
            return Vec4f::from_scalar(1.0) / pow(vec, -n);
        }
        let mut answer = Vec4f::from_scalar(1.0);
        loop {
            if (n & 1) > 0 {
                answer *= vec;
            }
            n >>= 1;
            if n == 0 {
                break answer;
            }
            vec *= vec;
        }
    }

    //TODO - return original value instead of NaN
    pub fn round(vec: Vec4f) -> Vec4f {
        #[cfg(target_feature = "sse4.1")] {
            Vec4f {
                xmm : unsafe { _mm_round_ps(vec.xmm, 8) }
            }
        }
        #[cfg(not(target_feature = "sse4.1"))] {
            Vec4f {
                xmm : unsafe { _mm_cvtepi32_ps(_mm_cvtps_epi32(vec.xmm)) }
            }
        }
    }

    //TODO - return original value instead of NaN
    pub fn truncate(vec: Vec4f) -> Vec4f {
        #[cfg(target_feature = "sse4.1")] {
            Vec4f {
                xmm : unsafe { _mm_round_ps(vec.xmm, 3 + 8) }
            }
        }
        #[cfg(not(target_feature = "sse4.1"))] {
            Vec4f {
                xmm : unsafe { _mm_cvtepi32_ps(_mm_cvtps_epi32(vec.xmm)) }
            }
        }
    }

    pub fn approx_recipr(vec: Vec4f) -> Vec4f {
        Vec4f {
            xmm : unsafe { _mm_rcp_ps(vec.xmm) }
        }
    }

    pub fn approx_rsqrt(vec: Vec4f) -> Vec4f {
        Vec4f {
            xmm : unsafe { _mm_rsqrt_ps(vec.xmm) }
        }
    }
    /*
    #[derive(Clone, Debug, Copy)]
    pub struct Vec4fb {
        xmm: __m128,
    }

    impl Vec4fb {
        pub fn new(b0: bool, b1: bool, b2: bool, b3: bool) -> Self {
            Self {
                xmm: unsafe { _mm_castsi128_ps(_mm_setr_epi32(-(b0 as i32), -(b1 as i32), -(b2 as i32), -(b3 as i32)) }
            }
        }

        pub fn from_scalar(value: bool) -> Self {
            Self {
                xmm: unsafe { _mm_castsi128_ps(_mm_set1_epi32(-(value as i32))) }
            }
        }
    }
    */
}

#[cfg(
    not(all(
        any(
            target_arch = "x86",
            target_arch = "x86_64"
        ),
        target_feature="sse"
    ))
)]
pub mod vec128 {}
