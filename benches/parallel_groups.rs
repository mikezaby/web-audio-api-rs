// The iai job runs every bench target under valgrind, where this one would take minutes.
#![cfg_attr(feature = "iai", allow(dead_code))]

//! Renders the graph in `examples/parallel_groups/tracks.rs` for 10 seconds offline.
//!
//! Baseline on one thread, before any parallel work (`main` at `03d0cb2`): 3.73 s on an M1 Pro.
//!
//! `cargo bench --bench parallel_groups -- --save-baseline single-thread`, then after a change
//! `cargo bench --bench parallel_groups -- --baseline single-thread`.
//! `cargo run --release --example parallel_groups` plays the same graph in real time.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, Criterion};

use web_audio_api::context::OfflineAudioContext;

#[path = "../examples/parallel_groups/tracks.rs"]
mod tracks;

const SAMPLE_RATE: f32 = 48000.;
const DURATION: usize = 10;

fn render_tracks(grouped: bool) {
    let length = SAMPLE_RATE as usize * DURATION;
    let mut ctx = OfflineAudioContext::new(2, black_box(length), SAMPLE_RATE);
    tracks::build(&ctx, tracks::VOICES, grouped);
    assert_eq!(ctx.start_rendering_sync().length(), length);
}

fn bench_tracks(c: &mut Criterion) {
    c.bench_function("parallel_groups_8_tracks", |b| {
        b.iter(|| render_tracks(false))
    });
    c.bench_function("parallel_groups_8_tracks_grouped", |b| {
        b.iter(|| render_tracks(true))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(30));
    targets = bench_tracks
}

#[cfg(not(feature = "iai"))]
criterion::criterion_main!(benches);

#[cfg(feature = "iai")]
fn main() {}
