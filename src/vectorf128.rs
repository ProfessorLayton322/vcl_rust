//! This module contains `Vec4f` struct with methods and functions to work with it
//!
//! This crate can only be compiled on `x86` or `x86_64` architecture and a proccessor that supports at
//! least `SSE2` instruction set

use crate::intrinsics::*;

use std::option::Option;

fn selectf(s: __m128, a: __m128, b: __m128) -> __m128 {
    #[cfg(target_feature = "sse4.1")]
    {
        // SAFETY: sse4.1
        unsafe { _mm_blendv_ps(b, a, s) }
    }
    #[cfg(not(target_feature = "sse4.1"))]
    {
        // SAFETY: sse2
        unsafe { _mm_or_ps(_mm_and_ps(s, a), _mm_andnot_ps(s, b)) }
    }
}

/// Packed array of four `f32` values that can be used for SIMD operations
#[derive(Clone, Copy)]
pub struct Vec4f {
    xmm: __m128,
}

impl Vec4f {
    /// Associated const - size of the packed vector
    pub const LEN: usize = 4;

    /// Returns `Vec4f` that contains four `f32` values that are equal to the arguments
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(vec, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_setr_ps(a, b, c, d) },
        }
    }

    /// Returns `Vec4f` that contains four values of type `f32` equal to the argument
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::from_scalar(2.0);
    /// assert_eq!(vec, [2.0f32; 4]);
    /// ```
    pub fn from_scalar(value: f32) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_set1_ps(value) },
        }
    }

    /// Squares every value of the vector and returns a copy
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(vec.squared(), [1.0, 4.0, 9.0, 16.0]);
    /// ```
    pub fn squared(self) -> Self {
        self * self
    }

    fn nan_vec() -> Vec4f {
        //These are magic numbers from original Agner Fog's lib
        //https://github.com/vectorclass/version2/blob/master/instrset.h#L415
        Vec4f::from_scalar(f32::from_bits(0x7FC00000 | (0x100 & 0x003FFFFF)))
    }

    /// Raises every every value of the vector and returns a copy
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(vec.pow(3), [1.0, 8.0, 27.0, 64.0]);
    /// ```
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
            if n % 2 > 0 {
                answer *= power;
            }
            n /= 2;
            if n == 0 {
                break answer;
            }
            power *= power;
        }
    }

    //TODO unchecked options without panic
    /// Copies values of the vector to a mutable slice
    ///
    /// # Panics
    ///
    /// Panics if `buffer.len()` is less than 4
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// let mut arr = [0.0f32; 4];
    /// vec.store(&mut arr);
    /// assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn store(self, buffer: &mut [f32]) {
        if buffer.len() < 4 {
            panic!("Buffer len not enough to store Vec128f");
        }
        // SAFETY: sse
        unsafe { _mm_storeu_ps(buffer.as_mut_ptr(), self.xmm) }
    }

    /// Copies values of the vector to a mutable slice. Address of the slice must be divisible by
    /// `16`. Is more efficient than `store`
    ///
    ///  # Panics
    ///
    /// Panics if `buffer.len()` is less than `4` or the address of the slice is not aligned
    /// by `16` bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// let mut arr = [0.0f32; 7];
    /// let diff = (arr.as_ptr() as usize % 16) / 4;
    /// let aligned = match diff {
    ///     0 => &mut arr[..4],
    ///     _ => &mut arr[4 - diff..8 - diff],
    /// };
    /// vec.store_aligned(aligned);
    /// assert_eq!(aligned, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn store_aligned(self, buffer: &mut [f32]) {
        if buffer.len() < 4 {
            panic!("Buffer len not enough to store Vec128f");
        }
        //check if buffer address is divisible by 16
        if (buffer.as_ptr() as usize) & 0xf > 0 {
            panic!("Buffer address is not aligned by 16");
        }
        // SAFETY: sse
        unsafe { _mm_store_ps(buffer.as_mut_ptr(), self.xmm) }
    }

    /// Copies values of the vector to a mutable aligned slice without using cache. Address of the slice must be divisible by
    /// `16`. Might be more efficient than `store_aligned` for copying very large arrays of data
    ///
    ///  # Panics
    ///
    /// Panics if `buffer.len()` is less than `4` or the address of the slice is not aligned
    /// by `16` bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// let mut arr = [0.0f32; 7];
    /// let diff = (arr.as_ptr() as usize % 16) / 4;
    /// let aligned = match diff {
    ///     0 => &mut arr[..4],
    ///     _ => &mut arr[4 - diff..8 - diff],
    /// };
    /// vec.store_aligned_nocache(aligned);
    /// assert_eq!(aligned, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn store_aligned_nocache(self, buffer: &mut [f32]) {
        if buffer.len() < 4 {
            panic!("Buffer len not enough to store Vec128f");
        }
        //check if buffer address is divisible by 16
        if (buffer.as_ptr() as usize) & 0xf > 0 {
            panic!("Buffer address is not aligned by 16");
        }
        // SAFETY: sse
        unsafe { _mm_stream_ps(buffer.as_mut_ptr(), self.xmm) }
    }

    /// Copies values of the vector to a mutable slice. Works for slices with size less than `4`
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// let mut arr = [0.0f32; 3];
    /// vec.store_partial(&mut arr);
    /// assert_eq!(arr, [1.0, 2.0, 3.0]);
    /// ```
    pub fn store_partial(self, buffer: &mut [f32]) {
        if buffer.len() >= 4 {
            self.store(buffer);
            return;
        }
        let mut values = [0.0f32; 4];
        self.store(&mut values);
        buffer.copy_from_slice(&values[..buffer.len()]);
    }

    /// Loads values from float slice
    ///
    /// # Panics
    ///
    /// Panics if `buffer.len()` is less than `4`
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let arr: [f32; 4] = [-2.0, 1.0, 3.0, -4.0];
    /// let mut d = Vec4f::default();
    /// d.load(&arr);
    /// assert_eq!(d, [-2.0, 1.0, 3.0, -4.0]);
    /// ```
    pub fn load(&mut self, buffer: &[f32]) {
        if buffer.len() < 4 {
            panic!("Buffer len not enough to load verctor");
        }
        // SAFETY: sse
        self.xmm = unsafe { _mm_loadu_ps(buffer.as_ptr()) };
    }

    /// Loads values from float slice aligned by `16` bytes
    ///
    /// # Panics
    ///
    /// Panics if the `buffer.len()` is less than `4` or if it's address is not aligned by `16`
    /// bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let mut vec = Vec4f::default();
    /// let arr = [2.0f32; 7];
    /// let diff = (arr.as_ptr() as usize % 16) / 4;
    /// let aligned = match diff {
    ///     0 => &arr[..4],
    ///     _ => &arr[4 - diff..8 - diff],
    /// };
    /// vec.load_aligned(aligned);
    /// assert_eq!(aligned, [2.0, 2.0, 2.0, 2.0]);
    /// ```
    pub fn load_aligned(&mut self, buffer: &[f32]) {
        if buffer.len() < 4 {
            panic!("Buffer len not enough to store Vec128f");
        }
        //check if buffer address is divisible by 16
        if (buffer.as_ptr() as usize) & 0xf > 0 {
            panic!("Buffer address is not aligned by 16");
        }
        // SAFETY: sse
        self.xmm = unsafe { _mm_load_ps(buffer.as_ptr()) };
    }

    /// Copies values from `buffer` slice to the vector. If `buffer.len()` is less than `4`
    /// fills vector's tail with zeroes
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let arr = [-2.0, 1.0];
    /// let mut d = Vec4f::default();
    /// d.load_partial(&arr);
    /// assert_eq!(d, [-2.0, 1.0, 0.0, 0.0]);
    /// ```
    pub fn load_partial(&mut self, buffer: &[f32]) {
        //We can use get_unchecked because we know the size of the buffer
        match buffer.len() {
            0 => *self = Self::default(),
            // SAFETY: sse
            1 => self.xmm = unsafe { _mm_load_ss(buffer.as_ptr()) },
            // SAFETY: sse
            2 => {
                self.xmm = unsafe {
                    _mm_setr_ps(*buffer.get_unchecked(0), *buffer.get_unchecked(1), 0.0, 0.0)
                }
            }
            // SAFETY: sse
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
            _ => self.load(buffer),
        };
    }

    /// Inserts `f32` value to the chosen `index` and returns the modified vector
    ///
    /// # Panics
    ///
    /// Panics if index is greater than 3
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let mut a = Vec4f::default();
    /// a = a.insert(3, 23.0);
    /// assert_eq!(a, [0.0, 0.0, 0.0, 23.0]);
    /// a = a.insert(0, 6.0);
    /// assert_eq!(a, [6.0, 0.0, 0.0, 23.0]);
    /// ```
    pub fn insert(self, index: usize, value: f32) -> Self {
        if index > 3 {
            panic!("Index out of bounds");
        }
        #[cfg(target_feature = "sse4.1")]
        {
            // SAFETY: sse4.1
            Self {
                xmm: unsafe { _mm_insert_ps(self.xmm, _mm_set_ss(value), (index as i32) << 4) },
            }
        }
        #[cfg(not(target_feature = "sse4.1"))]
        {
            let maskl: [i32; 8] = [0, 0, 0, 0, -1, 0, 0, 0];
            //we can use .add because 4 - index is positive
            let float_mask: *const f32 = unsafe { maskl.as_ptr().add(4 - index).cast() };
            // SAFETY: sse
            let broad: __m128 = unsafe { _mm_set1_ps(value) };
            // SAFETY: sse
            let mask: __m128 = unsafe { _mm_loadu_ps(float_mask) };
            Self {
                xmm: selectf(mask, broad, self.xmm),
            }
        }
    }

    /// Returns reference to `f32` value by `index`
    ///
    /// # Safety
    ///
    /// Caller must ensure that `index` is less than 4
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(unsafe { *vec.get_unchecked(2) }, 3.0);
    /// ```
    pub unsafe fn get_unchecked(&self, index: usize) -> &f32 {
        let float_pointer: *const f32 = &self.xmm as *const __m128 as *const f32;
        //add(index) is used accounting to index < 4
        unsafe { float_pointer.add(index).as_ref().unwrap() }
    }

    /// Return reference to `f32` value by `index`. Returns `None` if `index` is greater than `3`
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(*vec.get(2).unwrap(), 3.0);
    /// ```
    pub fn get(&self, index: usize) -> Option<&f32> {
        if index > 3 {
            return None;
        }
        //We can use unsafe because we checked that index is in bounds
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Cuts vector to `size`, replaces all tail values by zeroes and returns the modified copy
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let e = Vec4f::new(-3.0, 2.0, 1.0, 11.0);
    /// assert_eq!(e.cutoff(2), [-3.0, 2.0, 0.0, 0.0]);
    /// ```
    pub fn cutoff(self, size: usize) -> Self {
        if size >= 4 {
            return self;
        }
        let maskl: [i32; 8] = [-1, -1, -1, -1, 0, 0, 0, 0];
        //we can use .add because 4 - size is positive
        let float_mask: *const f32 = unsafe { maskl.as_ptr().add(4 - size).cast() };
        // SAFETY: sse
        let mask: __m128 = unsafe { _mm_loadu_ps(float_mask) };
        // SAFETY: sse
        Self {
            xmm: unsafe { _mm_and_ps(self.xmm, mask) },
        }
    }

    /// Truncates all values towards zero and return modified copy
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 1.5, 1.9, 2.0);
    /// assert_eq!(vec.truncate(), [1.0, 1.0, 1.0, 2.0]);
    /// ```
    pub fn truncate(self) -> Self {
        #[cfg(target_feature = "sse4.1")] {
            // SAFETY: sse4.1
            Self {
                xmm : unsafe { _mm_round_ps(self.xmm, 3 + 8) },
            }
        }
        #[cfg(not(target_feature = "sse4.1"))] {
            // SAFETY: sse2
            Self {
                xmm : unsafe { _mm_cvtepi32_ps(_mm_cvttps_epi32(self.xmm)) },
            }
        }
    }

    /// Rounds all values to closest integer and returns modified copy
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 1.4, 1.5, 1.6);
    /// assert_eq!(vec.round(), [1.0, 1.0, 2.0, 2.0]);
    /// ```
    pub fn round(self) -> Self {
        #[cfg(target_feature = "sse4.1")]
        {
            // SAFETY: sse4.1
            Self {
                xmm: unsafe { _mm_round_ps(self.xmm, 8) },
            }
        }
        #[cfg(not(target_feature = "sse4.1"))]
        {
            // SAFETY: sse2
            Self {
                xmm: unsafe { _mm_cvtepi32_ps(_mm_cvtps_epi32(self.xmm)) },
            }
        }
    }

    /// Returns a vector containing square roots of all values of original vector
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(0.0, 1.0, 2.0, 3.0);
    /// assert_eq!(vec.sqrt(), [0.0, 1.0, f32::sqrt(2.0), f32::sqrt(3.0)]);
    /// ```
    pub fn sqrt(self) -> Self {
        Self {
            // SAFETY: sse2
            xmm: unsafe { _mm_sqrt_ps(self.xmm) },
        }
    }

    /// Fast approximate calculation of reciprocal (i.e. `1 / self`)
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// let approx = vec.approx_recipr();
    /// let recipr = Vec4f::from_scalar(1.0) / vec;
    ///
    /// let mut arr = [0.0f32; 4];
    /// let diff = approx - recipr;
    /// diff.store(&mut arr);
    ///
    /// for value in arr.iter() {
    ///     assert!(value.abs() < 0.01);
    /// }
    /// ```
    pub fn approx_recipr(self) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_rcp_ps(self.xmm) },
        }
    }

    /// Fast approximate of reverse square root (i.e. `1 / self.sqrt()`)
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// let approx = vec.approx_rsqrt();
    /// let rsqrt = Vec4f::from_scalar(1.0) / vec.sqrt();
    ///
    /// let mut arr = [0.0f32; 4];
    /// let diff = approx - rsqrt;
    /// diff.store(&mut arr);
    ///
    /// for value in arr.iter() {
    ///     assert!(value.abs() < 0.01);
    /// }
    /// ```
    pub fn approx_rsqrt(self) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_rsqrt_ps(self.xmm) },
        }
    }

    const fn mask_helper(i: bool) -> i32 {
        match i {
            true => i32::MIN,
            false => 0,
        }
    }

    /// Changes signs of vector values chosen by compile-time parameters, returns modified copy
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(-2.0, 3.0, 2.0, -1.0);
    /// assert_eq!(
    ///     vec.change_sign::<true, false, true, false>(),
    ///     [2.0, 3.0, -2.0, -1.0]
    /// );
    /// ```
    pub fn change_sign<const I0: bool, const I1: bool, const I2: bool, const I3: bool>(
        self,
    ) -> Self {
        if !(I0 | I1 | I2 | I3) {
            return self;
        }
        // SAFETY: sse2
        let mask: __m128i = unsafe {
            _mm_setr_epi32(
                Self::mask_helper(I0),
                Self::mask_helper(I1),
                Self::mask_helper(I2),
                Self::mask_helper(I3),
            )
        };
        Self {
            // SAFETY: sse2
            xmm: unsafe { _mm_xor_ps(self.xmm, _mm_castsi128_ps(mask)) },
        }
    }

    /// Returns a vector containing absolute values of the original vector
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(-2.0, 3.0, 2.0, -1.0);
    /// assert_eq!(vec.abs(), [2.0, 3.0, 2.0, 1.0]);
    /// ```
    pub fn abs(self) -> Self {
        // SAFETY: sse2
        let mask: __m128 = unsafe { _mm_castsi128_ps(_mm_set1_epi32(i32::MAX)) };
        // SAFETY: sse
        Self {
            xmm: unsafe { _mm_and_ps(self.xmm, mask) },
        }
    }

    /// Calculates the sum of all vector values
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(vec.horizontal_add(), 10.0);
    /// ```
    pub fn horizontal_add(self) -> f32 {
        #[cfg(target_feature = "sse3")]
        {
            // SAFETY: sse3
            unsafe {
                let t1: __m128 = _mm_hadd_ps(self.xmm, self.xmm);
                let t2: __m128 = _mm_hadd_ps(t1, t1);
                _mm_cvtss_f32(t2)
            }
        }
        #[cfg(not(target_feature = "sse3"))]
        {
            // SAFETY: sse
            unsafe {
                let t1: __m128 = _mm_movehl_ps(self.xmm, self.xmm);
                let t2: __m128 = _mm_add_ps(self.xmm, t1);
                let t3: __m128 = _mm_shuffle_ps(t2, t2, 1);
                let t4: __m128 = _mm_add_ss(t2, t3);
                _mm_cvtss_f32(t4)
            }
        }
    }

    /// Chooses maximum for each index from two vectors, returns the result
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let first = Vec4f::new(5.0, -2.0, 3.0, 1.0);
    /// let second = Vec4f::new(1.0, 2.0, 2.0, 11.0);
    /// assert_eq!(Vec4f::max(first, second), [5.0, 2.0, 3.0, 11.0]);
    /// ```
    pub fn max(first: Vec4f, second: Vec4f) -> Vec4f {
        Vec4f {
            // SAFETY: sse
            xmm: unsafe { _mm_max_ps(first.xmm, second.xmm) },
        }
    }

    /// Chooses minimum for each index from two vectors, returns the result
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// let first = Vec4f::new(5.0, -2.0, 3.0, 1.0);
    /// let second = Vec4f::new(1.0, 2.0, 2.0, 11.0);
    /// assert_eq!(Vec4f::min(first, second), [1.0, -2.0, 2.0, 1.0]);
    /// ```
    pub fn min(first: Vec4f, second: Vec4f) -> Vec4f {
        Vec4f {
            // SAFETY: sse
            xmm: unsafe { _mm_min_ps(first.xmm, second.xmm) },
        }
    }

    /// Returns value of `a` with sign inverted if `b` has it's sign bit set (including `-0.0f32`)
    ///
    /// # Examples
    ///
    /// ```
    /// use vcl_rust::Vec4f;
    ///
    /// assert_eq!(
    ///     Vec4f::sign_combine(
    ///         Vec4f::new(-2.0, -1.0, 0.0, 1.0),
    ///         Vec4f::new(-10.0, 0.0, -20.0, 30.0)
    ///     ),
    ///     [2.0, -1.0, -0.0, 1.0]
    /// );
    /// ```
    pub fn sign_combine(a: Vec4f, b: Vec4f) -> Vec4f {
        a ^ (b & Vec4f::from_scalar(-0.0f32))
    }
}


/// Constructs vector from array
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let vec = Vec4f::from([1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(vec, [1.0, 2.0, 3.0, 4.0]);
/// ```
impl std::convert::From<[f32; 4]> for Vec4f {
    fn from(value: [f32; 4]) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_loadu_ps(value.as_ptr()) },
        }
    }
}

/// Constructs vector from slice 
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let arr = vec![-1.0, 2.0, 3.0, 1.5];
/// let vec = Vec4f::from(arr.as_slice());
/// assert_eq!(vec, [-1.0, 2.0, 3.0, 1.5]);
/// ```
impl std::convert::From<&[f32]> for Vec4f {
    fn from(value: &[f32]) -> Self {
        if value.len() < 4 {
            panic!("Slice size is not enough to construct a vector");
        }
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_loadu_ps(value.as_ptr()) },
        }
    }
}

/// Creates vector initialized with `0.0` values
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let vec = Vec4f::default();
/// assert_eq!(vec, [0.0f32; 4]);
/// ```
impl std::default::Default for Vec4f {
    fn default() -> Self {
        Self::from_scalar(0.0)
    }
}

/// Sum of two vectors
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let a = Vec4f::new(1.0, 2.0, 3.0, 4.0);
/// let b = Vec4f::new(1.0, 2.0, 3.0, 4.0);
/// assert_eq!(a * b, [1.0, 4.0, 9.0, 16.0]);
/// ```
impl std::ops::Add for Vec4f {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_add_ps(self.xmm, other.xmm) },
        }
    }
}

/// Add one vector to another and assign
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let mut a = Vec4f::new(1.0, 3.0, 2.0, -2.0);
/// a += Vec4f::new(-1.0, 2.0, 0.5, -3.0);
/// assert_eq!(a, [0.0, 5.0, 2.5, -5.0]);
/// ```
impl std::ops::AddAssign for Vec4f {
    fn add_assign(&mut self, other: Self) {
        // SAFETY: sse
        self.xmm = unsafe { _mm_add_ps(self.xmm, other.xmm) }
    }
}

/// Difference of two vectors
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let a = Vec4f::new(1.0, 2.0, 3.0, 4.0);
/// let b = Vec4f::new(2.0, 1.0, 4.0, 5.0);
/// assert_eq!(a - b, [-1.0, 1.0, -1.0, -1.0]);
/// ```
impl std::ops::Sub for Vec4f {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_sub_ps(self.xmm, other.xmm) },
        }
    }
}

/// Substract one vector from another and assign
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let mut a = Vec4f::new(1.0, 3.0, 2.0, -2.0);
/// a -= Vec4f::new(-1.0, 2.0, 0.5, -3.0);
/// assert_eq!(a, [2.0, 1.0, 1.5, 1.0]);
/// ```
impl std::ops::SubAssign for Vec4f {
    fn sub_assign(&mut self, other: Self) {
        // SAFETY: sse
        self.xmm = unsafe { _mm_sub_ps(self.xmm, other.xmm) }
    }
}

/// Negate vector
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let a = Vec4f::from_scalar(2.0);
/// assert_eq!(-a, [-2.0f32; 4]);
/// ```
impl std::ops::Neg for Vec4f {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            // SAFETY: sse2
            xmm: unsafe { _mm_xor_ps(self.xmm, _mm_castsi128_ps(_mm_set1_epi32(i32::MIN))) },
        }
    }
}

/// Multiply two vectors
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let a = Vec4f::new(1.0, 2.0, 3.0, 4.0);
/// let b = Vec4f::new(2.0, 1.0, 4.0, 5.0);
/// assert_eq!(a * b, [2.0, 2.0, 12.0, 20.0]);
/// ```
impl std::ops::Mul for Vec4f {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_mul_ps(self.xmm, other.xmm) },
        }
    }
}

/// Multiply one vector by another and assign
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let mut a = Vec4f::new(1.0, 3.0, 2.0, -2.0);
/// a *= Vec4f::new(-1.0, 2.0, 0.5, -3.0);
/// assert_eq!(a, [-1.0, 6.0, 1.0, 6.0]);
/// ```
impl std::ops::MulAssign for Vec4f {
    fn mul_assign(&mut self, other: Self) {
        // SAFETY: sse
        self.xmm = unsafe { _mm_mul_ps(self.xmm, other.xmm) }
    }
}

/// Divide one vector by another
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let a = Vec4f::new(3.0, 2.0, 10.0, -3.0);
/// let b = Vec4f::new(-1.0, 2.0, -2.0, -3.0);
/// assert_eq!(a / b, [-3.0, 1.0, -5.0, 1.0]);
/// ```
impl std::ops::Div for Vec4f {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_div_ps(self.xmm, other.xmm) },
        }
    }
}

/// Divide one vector by another and assign
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let mut a = Vec4f::new(3.0, 2.0, 10.0, -3.0);
/// a /= Vec4f::new(-1.0, 2.0, -2.0, -3.0);
/// assert_eq!(a, [-3.0, 1.0, -5.0, 1.0]);
/// ```
impl std::ops::DivAssign for Vec4f {
    fn div_assign(&mut self, other: Self) {
        // SAFETY: sse
        self.xmm = unsafe { _mm_div_ps(self.xmm, other.xmm) }
    }
}

/// Bitwise AND of two vectors
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let a = Vec4f::new(2.0, 4.0, 12.0, -1.0);
/// let b = Vec4f::new(3.0, 6.0, 8.0, 3.0);
/// assert_eq!(a & b, [2.0, 4.0, 8.0, 0.0]);
/// ```
impl std::ops::BitAnd for Vec4f {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_and_ps(self.xmm, other.xmm) },
        }
    }
}

/// Bitwise AND and assign
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let mut a = Vec4f::new(2.0, 4.0, 12.0, -1.0);
/// a &= Vec4f::new(3.0, 6.0, 8.0, 3.0);
/// assert_eq!(a, [2.0, 4.0, 8.0, 0.0]);
/// ```
impl std::ops::BitAndAssign for Vec4f {
    fn bitand_assign(&mut self, other: Self) {
        // SAFETY: sse
        self.xmm = unsafe { _mm_and_ps(self.xmm, other.xmm) }
    }
}

/// Bitwise OR of two vectors
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let a = Vec4f::new(2.0, 4.0, 8.0, 3.0);
/// let b = Vec4f::new(3.0, 2.0, 4.0, 2.0);
/// assert_eq!(a | b, [3.0, 4.0, 16.0, 3.0]);
/// ```
impl std::ops::BitOr for Vec4f {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_or_ps(self.xmm, other.xmm) },
        }
    }
}

/// Bitwise OR and assign
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let mut a = Vec4f::new(2.0, 4.0, 8.0, 3.0);
/// a |= Vec4f::new(3.0, 2.0, 4.0, 2.0);
/// assert_eq!(a, [3.0, 4.0, 16.0, 3.0]);
/// ```
impl std::ops::BitOrAssign for Vec4f {
    fn bitor_assign(&mut self, other: Self) {
        // SAFETY: sse
        self.xmm = unsafe { _mm_or_ps(self.xmm, other.xmm) }
    }
}

/// Bitwise XOR of two vectors
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let a = Vec4f::new(2.0, 4.0, 8.0, 3.0);
/// let b = Vec4f::new(2.0, 4.0, 8.0, 3.0);
/// assert_eq!(a ^ b, [0.0, 0.0, 0.0, 0.0]);
/// ```
impl std::ops::BitXor for Vec4f {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self {
        Self {
            // SAFETY: sse
            xmm: unsafe { _mm_xor_ps(self.xmm, other.xmm) },
        }
    }
}

/// Bitwise XOR and assign
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let mut a = Vec4f::new(2.0, 4.0, 12.0, -1.0);
/// a ^= Vec4f::new(2.0, 4.0, 12.0, -1.0);
/// assert_eq!(a, [0.0f32; 4]);
/// ```
impl std::ops::BitXorAssign for Vec4f {
    fn bitxor_assign(&mut self, other: Self) {
        // SAFETY: sse
        self.xmm = unsafe { _mm_xor_ps(self.xmm, other.xmm) }
    }
}

/// Comparison
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let a = Vec4f::new(2.0, 4.0, 12.0, -1.0);
/// let b = Vec4f::new(2.0, 4.0, 12.0, -1.0);
/// assert_eq!(a, b);
///
/// let c = Vec4f::new(-1.0, 2.0, 2.5, 3.0);
/// assert_ne!(a, c);
/// ```
impl std::cmp::PartialEq for Vec4f {
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: sse
        let comparison : i32 = unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.xmm, other.xmm)) };
        comparison == 0x0Fi32
    }
}

/// Operator ==, compares vector to [f32; 4]
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
/// assert_eq!(vec, [1.0, 2.0, 3.0, 4.0]);
/// ```
impl std::cmp::PartialEq<[f32; 4]> for Vec4f {
    fn eq(&self, other: &[f32; 4]) -> bool {
        self.eq(&Vec4f::from(other as &[f32]))
    }
}

/// Operator []. Returns reference to vector element for `index` that is not greater than `3`
///
///  # Panics
///
///  Panics if `index` is greater than `3`
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
/// assert_eq!(vec[2], 3.0);
/// ```
impl std::ops::Index<usize> for Vec4f {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        if index > 3 {
            panic!("Index out of bounds");
        }
        //get_unchecked can be used because index is checked
        unsafe { self.get_unchecked(index) }
    }
}

/// Reinterprets vector as `[f32; 4]` and formats it as a debug string
///
/// # Examples
///
/// ```
/// use vcl_rust::Vec4f;
///
/// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
/// ```
impl std::fmt::Debug for Vec4f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0.0f32; 4];
        self.store(&mut arr);
        arr.fmt(f)
    }
}
