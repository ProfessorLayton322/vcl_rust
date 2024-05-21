//! Rust version of Agner Fog's [vectorclass lib](https://github.com/vectorclass/version2)
//!
//! This crate contains a struct that containts four packed `f32` values and uses SIMD instructions
//! to work with them
//!
//! This crate can only be compiled on `86` or `x86_64` architecture and a proccessor that supports at
//! least `sse2` instruction set
//!
//! This crate also has `no_std` attribute

#![no_std]

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
compile_error!("This crate is only supported for x86 and x86_64 architecture");

#[cfg(not(target_feature = "sse2"))]
compile_error!("This crate requires sse2 to be compiled");

#[cfg(target_arch = "x86")]
use core::arch::x86 as intrinsics;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as intrinsics;

//Only compiled on x86/x86_64 with sse2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
))]
mod vectorf128;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
))]
pub use vectorf128::Vec4f;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
    test
))]
mod tests {
    use crate::Vec4f;

    #[test]
    fn test_basic() {
        assert_eq!(Vec4f::LEN, 4);

        let a = Vec4f::from_scalar(32.0);
        let b = Vec4f::new(1.0, 2.0, 3.0, 4.0);
        let c = a + b;
        assert_eq!(c, [33.0, 34.0, 35.0, 36.0]);

        let arr: [f32; 4] = [-2.0, 1.0, 3.0, -4.0];
        let mut d = Vec4f::default();
        d.load(&arr);
        assert_eq!(d, [-2.0, 1.0, 3.0, -4.0]);

        let e = Vec4f::new(2.0, 4.0, 12.0, -1.0);
        let f = Vec4f::new(2.0, 4.0, 12.0, -1.0);
        assert_eq!(e, f);

        let g = Vec4f::new(-1.0, 2.0, 2.5, 3.0);
        assert_ne!(e, g);
    }

    #[test]
    fn test_arithmetic() {
        let mut a = Vec4f::default();
        a += Vec4f::new(1.0, -1.0, 2.0, 3.0);
        assert_eq!(a, [1.0, -1.0, 2.0, 3.0]);
        a *= Vec4f::new(3.0, -2.0, 5.0, -1.0);
        assert_eq!(a, [3.0, 2.0, 10.0, -3.0]);
        a /= Vec4f::new(-1.0, 2.0, -2.0, -3.0);
        assert_eq!(a, [-3.0, 1.0, -5.0, 1.0]);

        let b = a + Vec4f::new(1.0, 2.0, 1.0, 2.0);
        assert_eq!(b, [-2.0, 3.0, -4.0, 3.0]);
        let c = Vec4f::new(1.0, -2.0, 7.0, -3.0) - b;
        assert_eq!(c, [3.0, -5.0, 11.0, -6.0]);
        let d = Vec4f::new(2.0, -1.0, -2.0, 4.0) * c;
        assert_eq!(d, [6.0, 5.0, -22.0, -24.0]);
        let e = d / Vec4f::new(3.0, -5.0, -11.0, -3.0);
        assert_eq!(e, [2.0, -1.0, 2.0, 8.0]);

        assert_eq!(e.horizontal_add(), 11.0);

        let first = Vec4f::new(5.0, -2.0, 3.0, 1.0);
        let second = Vec4f::new(1.0, 2.0, 2.0, 11.0);
        assert_eq!(Vec4f::max(first, second), [5.0, 2.0, 3.0, 11.0]);
        assert_eq!(Vec4f::min(first, second), [1.0, -2.0, 2.0, 1.0]);

        let third = Vec4f::new(-2.0, 3.0, 2.0, -1.0);
        assert_eq!(third.abs(), [2.0, 3.0, 2.0, 1.0]);
        assert_eq!((-third), [2.0, -3.0, -2.0, 1.0]);

        assert_eq!(
            third.change_sign::<true, false, true, false>(),
            [2.0, 3.0, -2.0, -1.0]
        );

        assert_eq!(
            Vec4f::sign_combine(
                Vec4f::new(-2.0, -1.0, 0.0, 1.0),
                Vec4f::new(-10.0, 0.0, -20.0, 30.0)
            ),
            [2.0, -1.0, -0.0, 1.0]
        );

        assert_eq!(
            Vec4f::new(0.0, 1.0, 2.0, 3.0).sqrt(),
            [0.0, 1.0, f32::sqrt(2.0), f32::sqrt(3.0)]
        );

        assert_eq!(
            Vec4f::new(-1.0, 1.0, 2.0, 3.0).squared(),
            [1.0, 1.0, 4.0, 9.0]
        );

        assert_eq!(
            Vec4f::new(-1.0, 2.0, 3.0, 1.5).pow(2),
            [1.0, 4.0, 9.0, 1.5 * 1.5]
        );

        assert_eq!(
            Vec4f::new(-1.0, 2.0, 3.0, 1.5).pow(-1),
            [-1.0, 0.5, 1.0 / 3.0, 1.0 / 1.5]
        );

        assert_eq!(Vec4f::new(1.0, 1.4, 1.5, 1.6).round(), [1.0, 1.0, 2.0, 2.0]);

        assert_eq!(Vec4f::new(1.0, 1.5, 1.9, 2.0).truncate(), [1.0, 1.0, 1.0, 2.0]);
    }

    fn compare_approx_vec4f(vec: &Vec4f, expected: [f32; 4]) {
        let mut arr = [0.0f32; 4];
        vec.store(&mut arr);
        for i in 0..4 {
            assert!(f32::abs(arr[i] - expected[i]) < 0.001f32);
        }
    }

    #[test]
    fn test_approx() {
        let a = Vec4f::new(1.0, 2.0, 3.0, 4.0);
        compare_approx_vec4f(&a.approx_recipr(), [1.0, 0.5, 1.0 / 3.0, 1.0 / 4.0]);
        compare_approx_vec4f(
            &a.approx_rsqrt(),
            [
                1.0,
                1.0 / f32::sqrt(2.0),
                1.0 / f32::sqrt(3.0),
                1.0 / f32::sqrt(4.0),
            ],
        );
    }

    #[test]
    fn test_bitwise() {
        let mut a = Vec4f::new(2.0, 4.0, 12.0, -1.0);
        a &= Vec4f::new(3.0, 6.0, 8.0, 3.0);
        assert_eq!(a, [2.0, 4.0, 8.0, 0.0]);

        assert_eq!(
            (Vec4f::new(2.0, 4.0, 12.0, -1.0) & Vec4f::new(3.0, 6.0, 8.0, 3.0)),
            [2.0, 4.0, 8.0, 0.0]
        );

        a = Vec4f::new(2.0, 4.0, 8.0, 3.0);
        a |= Vec4f::new(3.0, 2.0, 4.0, 2.0);
        assert_eq!(a, [3.0, 4.0, 16.0, 3.0]);

        assert_eq!(
            (Vec4f::new(2.0, 4.0, 8.0, 3.0) | Vec4f::new(3.0, 2.0, 4.0, 2.0)),
            [3.0, 4.0, 16.0, 3.0]
        );

        a = Vec4f::new(2.0, 5.0, 7.0, -2.0);
        a ^= Vec4f::new(2.0, 5.0, 7.0, -2.0);
        assert_eq!(a, [0.0, 0.0, 0.0, 0.0]);

        assert_eq!(
            (Vec4f::new(2.0, 4.0, 8.0, 3.0) ^ Vec4f::new(2.0, 4.0, 8.0, 3.0)),
            [0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_insert_get() {
        let mut a = Vec4f::default();
        a = a.insert(0, 6.0);
        a = a.insert(1, 7.0);
        a = a.insert(2, -2.0);
        a = a.insert(3, 23.0);
        assert_eq!(a, [6.0, 7.0, -2.0, 23.0]);
        a = a.insert(2, 5.0);
        assert_eq!(a, [6.0, 7.0, 5.0, 23.0]);

        assert_eq!(*a.get(0).unwrap(), 6.0);
        assert_eq!(*a.get(1).unwrap(), 7.0);
        assert_eq!(*a.get(2).unwrap(), 5.0);
        assert_eq!(*a.get(3).unwrap(), 23.0);
        assert_eq!(unsafe { *a.get_unchecked(2) }, 5.0);
        assert!(a.get(4).is_none());

        assert_eq!(a[0], 6.0);
        assert_eq!(a[1], 7.0);
        assert_eq!(a[2], 5.0);
        assert_eq!(a[3], 23.0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_insert_panic() {
        let a = Vec4f::default();
        a.insert(4, 123.0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_panic() {
        let a = Vec4f::default();
        a[5];
    }

    //Selects aligned buffer
    fn select_aligned(buffer: &mut [f32; 7]) -> &mut [f32] {
        let diff = (buffer.as_ptr() as usize % 16) / 4;
        if diff == 0 {
            return &mut buffer[..4];
        }
        let return_item = &mut buffer[(4 - diff)..(8 - diff)];
        assert_eq!(return_item.as_ptr() as usize % 16, 0);
        return_item
    }

    #[test]
    fn test_load_store() {
        let arr: [f32; 4] = [-2.0, 1.0, 4.0, 5.0];
        let mut a = Vec4f::default();
        a.load(&arr);
        assert_eq!(a, [-2.0, 1.0, 4.0, 5.0]);

        let some_array: [f32; 4] = [-1.0, 2.0, 3.0, -2.0];

        let mut unaligned = [0.0f32; 7];
        let aligned = select_aligned(&mut unaligned);
        aligned.copy_from_slice(&some_array);

        let mut b = Vec4f::default();
        b.load_aligned(aligned);
        assert_eq!(b, some_array);

        b.store_aligned(aligned);
        assert_eq!(aligned, some_array);

        b.store_aligned_nocache(aligned);
        assert_eq!(aligned, some_array);

        let mut buffer = [0.0f32; 4];
        b.store_partial(&mut buffer[..1]);
        assert_eq!(buffer, [-1.0, 0.0, 0.0, 0.0]);
        b.store_partial(&mut buffer[..2]);
        assert_eq!(buffer, [-1.0, 2.0, 0.0, 0.0]);
        b.store_partial(&mut buffer[..3]);
        assert_eq!(buffer, [-1.0, 2.0, 3.0, 0.0]);
        b.store_partial(&mut buffer);
        assert_eq!(buffer, [-1.0, 2.0, 3.0, -2.0]);

        let another_arr: [f32; 4] = [-10.0, 3.0, -2.0, 7.0];
        let mut c = Vec4f::default();

        c.load_partial(&another_arr[0..0]);
        assert_eq!(c, [0.0, 0.0, 0.0, 0.0]);
        c.load_partial(&another_arr[..1]);
        assert_eq!(c, [-10.0, 0.0, 0.0, 0.0]);
        c.load_partial(&another_arr[..2]);
        assert_eq!(c, [-10.0, 3.0, 0.0, 0.0]);
        c.load_partial(&another_arr[..3]);
        assert_eq!(c, [-10.0, 3.0, -2.0, 0.0]);
        c.load_partial(&another_arr);
        assert_eq!(c, [-10.0, 3.0, -2.0, 7.0]);

        let too_large_array = [0.0f32; 5];

        let mut d = Vec4f::new(-3.0, 2.0, 1.0, 11.0);
        d.load_partial(&too_large_array);
        assert_eq!(d, [0.0f32; 4]);

        let e = Vec4f::new(-3.0, 2.0, 1.0, 11.0);
        assert_eq!(e.cutoff(3), [-3.0, 2.0, 1.0, 0.0]);
        assert_eq!(e.cutoff(2), [-3.0, 2.0, 0.0, 0.0]);
        assert_eq!(e.cutoff(1), [-3.0, 0.0, 0.0, 0.0]);
        assert_eq!(e.cutoff(0), [0.0, 0.0, 0.0, 0.0]);
    }
}
