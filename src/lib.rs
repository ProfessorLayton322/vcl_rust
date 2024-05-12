mod vectorf128;
mod vectorf128e;

use crate::vectorf128::vec128::*;
use crate::vectorf128e::vec128e::*;

#[test]
fn test_basic() {
    assert_eq!(Vec4f::size(), 4);

    let a = Vec4f::from_scalar(32.0);
    let b = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    let c = a + b;
    assert_eq!(c, [33.0, 34.0, 35.0, 36.0]);

    let arr : [f32; 4] = [-2.0, 1.0, 3.0, -4.0];
    let mut d = Vec4f::default();
    d.load(arr.as_ptr());
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

    assert_eq!(square(Vec4f::new(-1.0, 1.0, 2.0, 3.0)),
        [1.0, 1.0, 4.0, 9.0]);

    assert_eq!(pow(Vec4f::new(-1.0, 2.0, 3.0, 1.5), 2),
        [1.0, 4.0, 9.0, 1.5 * 1.5]);
    assert_eq!(pow(Vec4f::new(-1.0, 2.0, 3.0, 1.5), -1),
        [-1.0, 0.5, 1.0 / 3.0, 1.0 / 1.5]);

    assert_eq!(round(Vec4f::new(1.0, 1.4, 1.5, 1.6)),
        [1.0, 1.0, 2.0, 2.0]);
}

fn compare_approx_vec4f(vec: &Vec4f, expected : [f32; 4]) {
    let mut arr = [0.0f32; 4];
    vec.store(arr.as_mut_ptr());
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
fn test_insert_extract() {
    let mut a = Vec4f::default();
    a.insert(0, 6.0);
    a.insert(1, 7.0);
    a.insert(2, -2.0);
    a.insert(3, 23.0);
    assert_eq!(a, [6.0, 7.0, -2.0, 23.0]);
    a.insert(2, 5.0);
    assert_eq!(a, [6.0, 7.0, 5.0, 23.0]);

    assert_eq!(a.extract(0), 6.0);
    assert_eq!(a.extract(1), 7.0);
    assert_eq!(a.extract(2), 5.0);
    assert_eq!(a.extract(3), 23.0);
}

use aligned_array::{Aligned, A16};
#[test]
fn test_load_store() {
    let arr : [f32; 4] = [-2.0, 1.0, 4.0, 5.0];
    let mut a = Vec4f::default();
    a.load(arr.as_ptr());
    assert_eq!(a, [-2.0, 1.0, 4.0, 5.0]);

    let aligned : Aligned<A16, _> = Aligned([-1.0, 2.0, 3.0, -2.0]);
    assert!(aligned.as_ptr() as usize % 16 == 0);
    let mut b = Vec4f::default();
    b.load_a(aligned.as_ptr());
    assert_eq!(b, [-1.0, 2.0, 3.0, -2.0]);

    let mut aligned_buffer : Aligned<A16, _> = Aligned([0.0f32; 4]);
    assert!(aligned_buffer.as_ptr() as usize % 16 == 0);
    b.store_a(aligned_buffer.as_mut_ptr());
    assert_eq!(aligned, aligned_buffer);

    let mut another_aligned : Aligned<A16, _> = Aligned([0.0f32; 4]);
    assert!(another_aligned.as_ptr() as usize % 16 == 0);
    b.store_nt(another_aligned.as_mut_ptr());
    assert_eq!(aligned, another_aligned);

    let mut buffer = [0.0f32; 4];
    b.store_partial(1, buffer.as_mut_ptr());
    assert_eq!(buffer, [-1.0, 0.0, 0.0, 0.0]);
    b.store_partial(2, buffer.as_mut_ptr());
    assert_eq!(buffer, [-1.0, 2.0, 0.0, 0.0]);
    b.store_partial(3, buffer.as_mut_ptr());
    assert_eq!(buffer, [-1.0, 2.0, 3.0, 0.0]);
    b.store_partial(4, buffer.as_mut_ptr());
    assert_eq!(buffer, [-1.0, 2.0, 3.0, -2.0]);

    let another_arr : [f32; 4] = [-10.0, 3.0, -2.0, 7.0];
    let mut c = Vec4f::default();

    c.load_partial(0, another_arr.as_ptr());
    assert_eq!(c, [0.0, 0.0, 0.0, 0.0]);
    c.load_partial(1, another_arr.as_ptr());
    assert_eq!(c, [-10.0, 0.0, 0.0, 0.0]);
    c.load_partial(2, another_arr.as_ptr());
    assert_eq!(c, [-10.0, 3.0, 0.0, 0.0]);
    c.load_partial(3, another_arr.as_ptr());
    assert_eq!(c, [-10.0, 3.0, -2.0, 0.0]);
    c.load_partial(4, another_arr.as_ptr());
    assert_eq!(c, [-10.0, 3.0, -2.0, 7.0]);

    let mut d = Vec4f::new(-3.0, 2.0, 1.0, 11.0);
    d.load_partial(5, another_arr.as_ptr());
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
