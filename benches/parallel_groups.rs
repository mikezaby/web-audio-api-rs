// The iai job runs every bench target under valgrind, where this one would take minutes.
#![cfg_attr(feature = "iai", allow(dead_code))]

//! A heavy graph shaped like a Blibliki instrument, the yardstick for parallel rendering
//! (`docs/parallel-render.md`).
//!
//! Each track is independent up to the master gain, so each one can become its own render group.
//!
//! Baseline on one thread, before any parallel work (`main` at `03d0cb2`): 3.10 s on an M1 Pro.
//!
//! `cargo bench --bench parallel_groups -- --save-baseline single-thread`, then after a change
//! `cargo bench --bench parallel_groups -- --baseline single-thread`

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, Criterion};

use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, OscillatorType};

const SAMPLE_RATE: f32 = 48000.;
const DURATION: usize = 10;
const TRACKS: usize = 8;
const VOICES: usize = 16;
const IR_SECONDS: f32 = 1.;

fn render_tracks() {
    let length = SAMPLE_RATE as usize * DURATION;
    let mut ctx = OfflineAudioContext::new(2, black_box(length), SAMPLE_RATE);
    let end = DURATION as f64;

    let master = ctx.create_gain();
    master.gain().set_value(0.05);
    master.connect(&ctx.destination());

    // Deterministic decaying noise, a stand-in for a reverb impulse response.
    let ir_length = (IR_SECONDS * SAMPLE_RATE) as usize;
    let mut ir = ctx.create_buffer(2, ir_length, SAMPLE_RATE);
    for channel in 0..2 {
        for (i, sample) in ir.get_channel_data_mut(channel).iter_mut().enumerate() {
            let noise = (((i * 2 + channel) as f32 * 12.9898).sin() * 43758.547).fract();
            *sample = noise * (-6. * i as f32 / ir_length as f32).exp();
        }
    }

    for track in 0..TRACKS {
        // Stage A: `ctx.set_render_group(track as u32 + 1)` goes here.
        let bus = ctx.create_gain();
        let mut reverb = ctx.create_convolver();
        reverb.set_buffer(ir.clone());
        bus.connect(&master);
        bus.connect(&reverb);
        reverb.connect(&master);

        for voice in 0..VOICES {
            let mut osc = ctx.create_oscillator();
            osc.set_type(OscillatorType::Sawtooth);
            osc.frequency()
                .set_value(55. * 2_f32.powf((track * VOICES + voice) as f32 / 12.));
            osc.detune().linear_ramp_to_value_at_time(50., end);

            // a-rate automation keeps the filter and the amp off their constant-param fast paths
            let filter = ctx.create_biquad_filter();
            filter.q().set_value(5.);
            filter.frequency().set_value(200.);
            filter
                .frequency()
                .exponential_ramp_to_value_at_time(8000., end);

            let amp = ctx.create_gain();
            amp.gain().set_value(0.);
            amp.gain().linear_ramp_to_value_at_time(1., end);

            osc.connect(&filter);
            filter.connect(&amp);
            amp.connect(&bus);
            osc.start();
        }
    }

    assert_eq!(ctx.start_rendering_sync().length(), length);
}

fn bench_tracks(c: &mut Criterion) {
    c.bench_function("parallel_groups_8_tracks", |b| b.iter(render_tracks));
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
