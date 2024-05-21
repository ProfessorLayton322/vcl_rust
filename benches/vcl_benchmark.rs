use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vcl_rust::Vec4f;

fn add(n: u32) -> Vec4f {
    let a = Vec4f::new(2.0, -1.0, 3.0, 4.0);
    let mut b = Vec4f::default();
    for _ in 0..n {
        b += a;
    }
    b
}

fn div(n: u32) -> Vec4f {
    let a = Vec4f::new(2.0, -1.0, 3.0, 4.0);
    let mut b = Vec4f::from_scalar(1.0);
    for _ in 0..n {
        b /= a;
    }
    b
}

fn mul(n: u32) -> Vec4f {
    let a = Vec4f::new(2.0, -1.0, 3.0, 4.0);
    let mut b = Vec4f::from_scalar(1.0);
    for _ in 0..n {
        b *= a;
    }
    b
}

fn pow(data: &mut [Vec4f]) {
    data.iter_mut().for_each(|vec| *vec = vec.pow(100));
}

fn sqrt(data: &mut [Vec4f]) {
    data.iter_mut().for_each(|vec| *vec = vec.sqrt());
}

fn horizontal(data: &[Vec4f]) -> f32 {
    data.iter().map(|vec| vec.horizontal_add()).sum()
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("add 10", |b| b.iter(|| add(black_box(10))));
    c.bench_function("div 10", |b| b.iter(|| div(black_box(10))));
    c.bench_function("mul 10", |b| b.iter(|| mul(black_box(10))));

    let data = vec![Vec4f::new(1.0, 2.0, 3.0, 4.0); 10];
    c.bench_function("pow 10", |b| b.iter(|| pow(black_box(&mut data.clone()))));
    c.bench_function("sqrt 10", |b| b.iter(|| sqrt(black_box(&mut data.clone()))));
    c.bench_function("horizontal 10", |b| {
        b.iter(|| horizontal(black_box(&data.clone())))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
