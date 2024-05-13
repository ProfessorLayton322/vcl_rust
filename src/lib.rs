#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64"
)))]
compile_error!("This crate is only supported for x86 and x86_64 architecture");

#[cfg(not(target_feature = "sse2"))]
compile_error!("This crate requires sse2 to be compiled");

mod vectorf128;
use crate::vectorf128::vec128::*;

//mod vectorf128e;
//use crate::vectorf128e::vec128e::*;

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_basic() {
        assert_eq!(Vec4f::LEN, 4);

        let a = Vec4f::from_scalar(32.0);
        let b = Vec4f::new(1.0, 2.0, 3.0, 4.0);
        let c = a + b;
        assert_eq!(c, [33.0, 34.0, 35.0, 36.0]);

        let arr : [f32; 4] = [-2.0, 1.0, 3.0, -4.0];
        let mut d = Vec4f::default();
        d.load(&arr);
        assert_eq!(d, [-2.0, 1.0, 3.0, -4.0]);
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

        assert_eq!(horizontal_add(e), 11.0);

        let first = Vec4f::new(5.0, -2.0, 3.0, 1.0);
        let second = Vec4f::new(1.0, 2.0, 2.0, 11.0);
        assert_eq!(max(first, second), [5.0, 2.0, 3.0, 11.0]);
        assert_eq!(min(first, second), [1.0, -2.0, 2.0, 1.0]);

        let third = Vec4f::new(-2.0, 3.0, 2.0, -1.0);
        assert_eq!(abs(third), [2.0, 3.0, 2.0, 1.0]);
        assert_eq!((-third), [2.0, -3.0, -2.0, 1.0]);

        assert_eq!(change_sign::<true, false, true, false>(third), 
            [2.0, 3.0, -2.0, -1.0]);

        assert_eq!(sign_combine(
            Vec4f::new(-2.0, -1.0, 0.0, 1.0),
            Vec4f::new(-10.0, 0.0, -20.0, 30.0)),
            [2.0, -1.0, -0.0, 1.0]);

        assert_eq!(sqrt(Vec4f::new(0.0, 1.0, 2.0, 3.0)),
            [0.0, 1.0, f32::sqrt(2.0), f32::sqrt(3.0)]);

        assert_eq!(Vec4f::new(-1.0, 1.0, 2.0, 3.0).squared(),
            [1.0, 1.0, 4.0, 9.0]);

        assert_eq!(Vec4f::new(-1.0, 2.0, 3.0, 1.5).pow(2),
            [1.0, 4.0, 9.0, 1.5 * 1.5]);
        assert_eq!(Vec4f::new(-1.0, 2.0, 3.0, 1.5).pow(-1),
            [-1.0, 0.5, 1.0 / 3.0, 1.0 / 1.5]);

        assert_eq!(round(Vec4f::new(1.0, 1.4, 1.5, 1.6)),
            [1.0, 1.0, 2.0, 2.0]);
    }

    fn compare_approx_vec4f(vec: &Vec4f, expected : [f32; 4]) {
        let mut arr = [0.0f32; 4];
        vec.store(&mut arr);
        for i in 0..4 {
            assert!(f32::abs(arr[i] - expected[i]) < 0.001f32);
        }
    }

    #[test]
    fn test_approx() {
        let a = Vec4f::new(1.0, 2.0, 3.0, 4.0);
        compare_approx_vec4f(&approx_recipr(a), [1.0, 0.5, 1.0 / 3.0, 1.0 / 4.0]);
        compare_approx_vec4f(&approx_rsqrt(a), [1.0, 1.0 / f32::sqrt(2.0), 1.0 / f32::sqrt(3.0), 1.0 / f32::sqrt(4.0)]);
    }

    #[test]
    fn test_bitwise() {
        let mut a = Vec4f::new(2.0, 4.0, 12.0, -1.0);
        a &= Vec4f::new(3.0, 6.0, 8.0, 3.0);
        assert_eq!(a, [2.0, 4.0, 8.0, 0.0]);

        assert_eq!((
            Vec4f::new(2.0, 4.0, 12.0, -1.0) & 
            Vec4f::new(3.0, 6.0, 8.0, 3.0)
        ),
        [2.0, 4.0, 8.0, 0.0]);

        a = Vec4f::new(2.0, 4.0, 8.0, 3.0);
        a |= Vec4f::new(3.0, 2.0, 4.0, 2.0);
        assert_eq!(a, [3.0, 4.0, 16.0, 3.0]);

        assert_eq!((
            Vec4f::new(2.0, 4.0, 8.0, 3.0) |
            Vec4f::new(3.0, 2.0, 4.0, 2.0)
        ),
        [3.0, 4.0, 16.0, 3.0]);

        a = Vec4f::new(2.0, 5.0, 7.0, -2.0);
        a ^= Vec4f::new(2.0, 5.0, 7.0, -2.0);
        assert_eq!(a, [0.0, 0.0, 0.0, 0.0]);

        assert_eq!((
            Vec4f::new(2.0, 4.0, 8.0, 3.0) ^
            Vec4f::new(2.0, 4.0, 8.0, 3.0)
        ),
        [0.0, 0.0, 0.0, 0.0]);

    }

    #[test]
    fn test_insert_get() {
        let mut a = Vec4f::default();
        a.insert(0, 6.0);
        a.insert(1, 7.0);
        a.insert(2, -2.0);
        a.insert(3, 23.0);
        assert_eq!(a, [6.0, 7.0, -2.0, 23.0]);
        a.insert(2, 5.0);
        assert_eq!(a, [6.0, 7.0, 5.0, 23.0]);

        assert_eq!(a.get(0), 6.0);
        assert_eq!(a.get(1), 7.0);
        assert_eq!(a.get(2), 5.0);
        assert_eq!(a.get(3), 23.0);
    }

    fn select_aligned(buffer: &mut [f32; 5]) -> &mut [f32] {
        if (buffer.as_ptr() as usize) % 16 == 0 {
            return &mut buffer[..4];
        }
        return &mut buffer[1..];
    }

    #[test]
    fn test_load_store() {
        let arr : [f32; 4] = [-2.0, 1.0, 4.0, 5.0];
        let mut a = Vec4f::default();
        a.load(&arr);
        assert_eq!(a, [-2.0, 1.0, 4.0, 5.0]);

        let some_array : [f32; 4] = [-1.0, 2.0, 3.0, -2.0];

        let mut unaligned = [0.0f32; 5];
        let mut aligned = select_aligned(&mut unaligned);
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

        let another_arr : [f32; 4] = [-10.0, 3.0, -2.0, 7.0];
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
        assert_eq!(d, [-3.0, 2.0, 1.0, 11.0]);
        d.cutoff(3);
        assert_eq!(d, [-3.0, 2.0, 1.0, 0.0]);
        d.cutoff(2);
        assert_eq!(d, [-3.0, 2.0, 0.0, 0.0]);
        d.cutoff(1);
        assert_eq!(d, [-3.0, 0.0, 0.0, 0.0]);
        d.cutoff(0);
        assert_eq!(d, [0.0, 0.0, 0.0, 0.0]);
    }
}
