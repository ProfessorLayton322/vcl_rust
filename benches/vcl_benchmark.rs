use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vcl_rust::vectorf128::*;

fn add(n: u32) {
    let a = Vec4f::new(2.0, -1.0, 3.0, 4.0);
    let mut b = Vec4f::default();
    for _ in 0..n {
        b += a;
    }
}

fn div(n: u32) {
    let a = Vec4f::new(2.0, -1.0, 3.0, 4.0);
    let mut b = Vec4f::from_scalar(1.0);
    for _ in 0..n {
        b /= a;
    }
}

fn mul(n: u32) {
    let a = Vec4f::new(2.0, -1.0, 3.0, 4.0);
    let mut b = Vec4f::from_scalar(1.0);
    for _ in 0..n {
        b *= a;
    }
}

fn pow(n: u32) {
    let a = Vec4f::new(2.0, -1.0, 3.0, 4.0);
    for _ in 0..n {
        a.pow(100);
    }
}

fn sqrt(n: u32) {
    let a = Vec4f::new(31.0, 15.0, 111.0, 87.0);
    for _ in 0..n {
        a.sqrt();
    }
}

fn horizontal(n: u32) {
    let a = Vec4f::new(31.0, 15.0, 111.0, 87.0);
    for _ in 0..n {
        a.horizontal_add();
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("add 10", |b| b.iter(|| add(black_box(10))));
    c.bench_function("div 10", |b| b.iter(|| div(black_box(10))));
    c.bench_function("mul 10", |b| b.iter(|| mul(black_box(10))));
    c.bench_function("pow 10", |b| b.iter(|| pow(black_box(10))));
    c.bench_function("sqrt 10", |b| b.iter(|| sqrt(black_box(10))));
    c.bench_function("horizontal 10", |b| b.iter(|| horizontal(black_box(10))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
