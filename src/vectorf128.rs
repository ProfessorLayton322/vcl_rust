//This file should only be included if sse2 feature is supported and architecture is x86(_64)
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
compile_error!(
    "Vector128 module is not supposed to be compiled on any architecture other than x86 or x86_64"
);

#[cfg(not(target_feature = "sse2"))]
compile_error!("Vector128 module requires sse2 feature to be compiled");

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn selectf(s: __m128, a: __m128, b: __m128) -> __m128 {
    #[cfg(target_feature = "sse4.1")]
    {
        //sse4.1
        unsafe { _mm_blendv_ps(b, a, s) }
    }
    #[cfg(not(target_feature = "sse4.1"))]
    {
        //sse2
        unsafe { _mm_or_ps(_mm_and_ps(s, a), _mm_andnot_ps(s, b)) }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Vec4f {
    xmm: __m128,
}

impl Vec4f {
    pub const LEN: usize = 4;

    pub fn from_scalar(value: f32) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_set1_ps(value) },
        }
    }

    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_setr_ps(a, b, c, d) },
        }
    }

    pub fn squared(self) -> Self {
        self * self
    }

    fn nan_vec() -> Vec4f {
        //These are magic numbers from original Agner Fog's lib
        //https://github.com/vectorclass/version2/blob/master/instrset.h#L415
        Vec4f::from_scalar(f32::from_bits(0x7FC00000 | (0x100 & 0x003FFFFF)))
    }

    pub fn pow(self, mut n: i32) -> Self {
        if n < 0 {
            if n == i32::MIN {
                return Self::nan_vec();
            }
            return Vec4f::from_scalar(1.0) / self.pow(-n);
        }
        let mut answer = Vec4f::from_scalar(1.0);
        let mut power = self;
        loop {
            if (n & 1) > 0 {
                answer *= power;
            }
            n >>= 1;
            if n == 0 {
                break answer;
            }
            power *= power;
        }
    }

    //TODO unchecked options without panic
    pub fn store(self, buffer: &mut [f32]) {
        if buffer.len() < 4 {
            panic!("Buffer len not enough to store Vec128f");
        }
        //sse
        unsafe { _mm_storeu_ps(buffer.as_mut_ptr(), self.xmm) }
    }

    pub fn store_aligned(self, buffer: &mut [f32]) {
        if buffer.len() < 4 {
            panic!("Buffer len not enough to store Vec128f");
        }
        //checl if buffer address is divisible by 16
        if (buffer.as_ptr() as usize) & 0xf > 0 {
            panic!("Buffer address is not aligned by 16");
        }
        //sse
        unsafe { _mm_store_ps(buffer.as_mut_ptr(), self.xmm) }
    }

    pub fn store_aligned_nocache(self, buffer: &mut [f32]) {
        if buffer.len() < 4 {
            panic!("Buffer len not enough to store Vec128f");
        }
        //checl if buffer address is divisible by 16
        if (buffer.as_ptr() as usize) & 0xf > 0 {
            panic!("Buffer address is not aligned by 16");
        }
        //sse
        unsafe { _mm_stream_ps(buffer.as_mut_ptr(), self.xmm) }
    }

    pub fn store_partial(self, buffer: &mut [f32]) {
        if buffer.len() >= 4 {
            self.store(buffer);
            return;
        }
        let mut values = [0.0f32; 4];
        self.store(&mut values);
        buffer.copy_from_slice(&values[..buffer.len()]);
    }

    pub fn load(&mut self, buffer: &[f32]) {
        //sse
        self.xmm = unsafe { _mm_loadu_ps(buffer.as_ptr()) };
    }

    pub fn load_aligned(&mut self, buffer: &[f32]) {
        //sse
        self.xmm = unsafe { _mm_load_ps(buffer.as_ptr()) };
    }

    pub fn load_partial(&mut self, buffer: &[f32]) {
        //We can use get_unchecked because we know the size of the buffer
        match buffer.len() {
            0 => *self = Self::default(),
            //sse
            1 => self.xmm = unsafe { _mm_load_ss(buffer.as_ptr()) },
            //sse
            2 => {
                self.xmm = unsafe {
                    _mm_setr_ps(*buffer.get_unchecked(0), *buffer.get_unchecked(1), 0.0, 0.0)
                }
            }
            //sse
            3 => {
                self.xmm = unsafe {
                    _mm_setr_ps(
                        *buffer.get_unchecked(0),
                        *buffer.get_unchecked(1),
                        *buffer.get_unchecked(2),
                        0.0,
                    )
                }
            }
            4 => self.load(buffer),
            _ => {}
        };
    }

    pub fn insert(&mut self, index: usize, value: f32) {
        #[cfg(target_feature = "sse4.1")]
        {
            //sse4.1
            self.xmm =
                unsafe { _mm_insert_ps(self.xmm, _mm_set_ss(value), ((index & 3) as i32) << 4) };
        }
        #[cfg(not(target_feature = "sse4.1"))]
        {
            let maskl: [i32; 8] = [0, 0, 0, 0, -1, 0, 0, 0];
            //we can use .add because 4 - index & 3 is positive
            let float_mask: *const f32 = unsafe { maskl.as_ptr().add(4 - (index & 3)).cast() };
            //sse
            let broad: __m128 = unsafe { _mm_set1_ps(value) };
            //sse
            let mask: __m128 = unsafe { _mm_loadu_ps(float_mask) };
            self.xmm = selectf(mask, broad, self.xmm);
        }
    }

    pub fn get(&self, index: usize) -> f32 {
        let mut buffer = [0.0f32; 4];
        self.store(&mut buffer);
        //We can use get_unchecked because index & 3 < 4
        unsafe { *buffer.get_unchecked(index & 3) }
    }

    pub fn cutoff(self, size: usize) -> Self {
        if size >= 4 {
            return self;
        }
        let maskl: [i32; 8] = [-1, -1, -1, -1, 0, 0, 0, 0];
        //we can use .add because 4 - size is positive
        let float_mask: *const f32 = unsafe { maskl.as_ptr().add(4 - size).cast() };
        //sse
        let mask: __m128 = unsafe { _mm_loadu_ps(float_mask) };
        //sse
        Self {
            xmm: unsafe { _mm_and_ps(self.xmm, mask) },
        }
    }

    /*
    //TODO - return original value instead of NaN
    pub fn truncate(self) -> Self {
        #[cfg(target_feature = "sse4.1")] {
            //sse4.1
            Self {
                xmm : unsafe { _mm_round_ps(self.xmm, 3 + 8) }
            }
        }
        #[cfg(not(target_feature = "sse4.1"))] {
            //sse2
            Self {
                xmm : unsafe { _mm_cvtepi32_ps(_mm_cvtps_epi32(self.xmm)) }
            }
        }
    }
    */

    //TODO - return original value instead of NaN
    pub fn round(self) -> Self {
        #[cfg(target_feature = "sse4.1")]
        {
            //sse4.1
            Self {
                xmm: unsafe { _mm_round_ps(self.xmm, 8) },
            }
        }
        #[cfg(not(target_feature = "sse4.1"))]
        {
            //sse2
            Self {
                xmm: unsafe { _mm_cvtepi32_ps(_mm_cvtps_epi32(self.xmm)) },
            }
        }
    }

    pub fn sqrt(self) -> Self {
        Self {
            //sse2
            xmm: unsafe { _mm_sqrt_ps(self.xmm) },
        }
    }

    pub fn approx_recipr(self) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_rcp_ps(self.xmm) },
        }
    }

    pub fn approx_rsqrt(self) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_rsqrt_ps(self.xmm) },
        }
    }

    const fn mask_helper(i: bool) -> i32 {
        match i {
            true => i32::MIN,
            false => 0,
        }
    }

    pub fn change_sign<const I0: bool, const I1: bool, const I2: bool, const I3: bool>(
        self,
    ) -> Self {
        if !(I0 | I1 | I2 | I3) {
            return self;
        }
        //sse2
        let mask: __m128i = unsafe {
            _mm_setr_epi32(
                Self::mask_helper(I0),
                Self::mask_helper(I1),
                Self::mask_helper(I2),
                Self::mask_helper(I3),
            )
        };
        Self {
            //sse2
            xmm: unsafe { _mm_xor_ps(self.xmm, _mm_castsi128_ps(mask)) },
        }
    }

    pub fn abs(self) -> Self {
        //sse2
        let mask: __m128 = unsafe { _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)) };
        //sse
        Self {
            xmm: unsafe { _mm_and_ps(self.xmm, mask) },
        }
    }

    pub fn horizontal_add(self) -> f32 {
        #[cfg(target_feature = "sse3")]
        {
            //sse3
            unsafe {
                let t1: __m128 = _mm_hadd_ps(self.xmm, self.xmm);
                let t2: __m128 = _mm_hadd_ps(t1, t1);
                _mm_cvtss_f32(t2)
            }
        }
        #[cfg(not(target_feature = "sse3"))]
        {
            //sse
            unsafe {
                let t1: __m128 = _mm_movehl_ps(self.xmm, self.xmm);
                let t2: __m128 = _mm_add_ps(self.xmm, t1);
                let t3: __m128 = _mm_shuffle_ps(t2, t2, 1);
                let t4: __m128 = _mm_add_ss(t2, t3);
                _mm_cvtss_f32(t4)
            }
        }
    }
}

impl std::default::Default for Vec4f {
    fn default() -> Self {
        Self::from_scalar(0.0)
    }
}

impl std::ops::Add for Vec4f {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_add_ps(self.xmm, other.xmm) },
        }
    }
}

impl std::ops::AddAssign for Vec4f {
    fn add_assign(&mut self, other: Self) {
        //sse
        self.xmm = unsafe { _mm_add_ps(self.xmm, other.xmm) }
    }
}

impl std::ops::Sub for Vec4f {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_sub_ps(self.xmm, other.xmm) },
        }
    }
}

impl std::ops::SubAssign for Vec4f {
    fn sub_assign(&mut self, other: Self) {
        //sse
        self.xmm = unsafe { _mm_sub_ps(self.xmm, other.xmm) }
    }
}

impl std::ops::Neg for Vec4f {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            //sse2
            xmm: unsafe { _mm_xor_ps(self.xmm, _mm_castsi128_ps(_mm_set1_epi32(i32::MIN))) },
        }
    }
}

impl std::ops::Mul for Vec4f {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_mul_ps(self.xmm, other.xmm) },
        }
    }
}

impl std::ops::MulAssign for Vec4f {
    fn mul_assign(&mut self, other: Self) {
        //sse
        self.xmm = unsafe { _mm_mul_ps(self.xmm, other.xmm) }
    }
}

impl std::ops::Div for Vec4f {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_div_ps(self.xmm, other.xmm) },
        }
    }
}

impl std::ops::DivAssign for Vec4f {
    fn div_assign(&mut self, other: Self) {
        //sse
        self.xmm = unsafe { _mm_div_ps(self.xmm, other.xmm) }
    }
}

impl std::ops::BitAnd for Vec4f {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_and_ps(self.xmm, other.xmm) },
        }
    }
}

impl std::ops::BitAndAssign for Vec4f {
    fn bitand_assign(&mut self, other: Self) {
        //sse
        self.xmm = unsafe { _mm_and_ps(self.xmm, other.xmm) }
    }
}

impl std::ops::BitOr for Vec4f {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_or_ps(self.xmm, other.xmm) },
        }
    }
}

impl std::ops::BitOrAssign for Vec4f {
    fn bitor_assign(&mut self, other: Self) {
        //sse
        self.xmm = unsafe { _mm_or_ps(self.xmm, other.xmm) }
    }
}

impl std::ops::BitXor for Vec4f {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self {
        Self {
            //sse
            xmm: unsafe { _mm_xor_ps(self.xmm, other.xmm) },
        }
    }
}

impl std::ops::BitXorAssign for Vec4f {
    fn bitxor_assign(&mut self, other: Self) {
        //sse
        self.xmm = unsafe { _mm_xor_ps(self.xmm, other.xmm) }
    }
}

impl std::cmp::PartialEq<[f32; 4]> for Vec4f {
    fn eq(&self, other: &[f32; 4]) -> bool {
        let mut arr = [0.0f32; 4];
        self.store(&mut arr);
        arr == *other
    }
}

pub fn max(first: Vec4f, second: Vec4f) -> Vec4f {
    Vec4f {
        //sse
        xmm: unsafe { _mm_max_ps(first.xmm, second.xmm) },
    }
}

pub fn min(first: Vec4f, second: Vec4f) -> Vec4f {
    Vec4f {
        //sse
        xmm: unsafe { _mm_min_ps(first.xmm, second.xmm) },
    }
}

pub fn sign_combine(a: Vec4f, b: Vec4f) -> Vec4f {
    a ^ (b & Vec4f::from_scalar(-0.0f32))
}
