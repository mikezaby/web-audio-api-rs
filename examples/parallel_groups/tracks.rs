//! A heavy graph shaped like a Blibliki instrument, the yardstick for parallel rendering
//! (`docs/parallel-render.md`). Shared by `benches/parallel_groups.rs` and this example.
//!
//! Each track is independent up to the master gain, so each one can become its own render group.

use web_audio_api::context::BaseAudioContext;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, OscillatorType};

pub const TRACKS: usize = 8;
pub const VOICES: usize = 24;
const IR_SECONDS: f32 = 1.;
const SCALE: [f32; 5] = [0., 3., 5., 7., 10.];

/// With `grouped`, each track is its own render group.
pub fn build(ctx: &impl BaseAudioContext, voices: usize, grouped: bool) {
    let sample_rate = ctx.sample_rate();

    let master = ctx.create_gain();
    master.gain().set_value(0.02);
    master.connect(&ctx.destination());

    // Deterministic decaying noise, a stand-in for a reverb impulse response.
    let ir_length = (IR_SECONDS * sample_rate) as usize;
    let mut ir = ctx.create_buffer(2, ir_length, sample_rate);
    for channel in 0..2 {
        for (i, sample) in ir.get_channel_data_mut(channel).iter_mut().enumerate() {
            let noise = (((i * 2 + channel) as f32 * 12.9898).sin() * 43758.547).fract();
            *sample = noise * (-6. * i as f32 / ir_length as f32).exp();
        }
    }

    for track in 0..TRACKS {
        if grouped {
            ctx.set_render_group(track as u32 + 1);
        }
        let bus = ctx.create_gain();
        let mut reverb = ctx.create_convolver();
        reverb.set_buffer(ir.clone());
        bus.connect(&master);
        bus.connect(&reverb);
        reverb.connect(&master);

        // The track LFO keeps every voice's params a-rate, off their constant-value fast paths.
        let mut lfo = ctx.create_oscillator();
        lfo.frequency().set_value(0.2 + track as f32 * 0.07);
        let cutoff_depth = ctx.create_gain();
        cutoff_depth.gain().set_value(2000.);
        let amp_depth = ctx.create_gain();
        amp_depth.gain().set_value(0.5);
        let vibrato_depth = ctx.create_gain();
        vibrato_depth.gain().set_value(10.);
        lfo.connect(&cutoff_depth);
        lfo.connect(&amp_depth);
        lfo.connect(&vibrato_depth);
        lfo.start();

        for voice in 0..voices {
            let step = track * voices + voice;
            let semitones = (step / SCALE.len() % 4 * 12) as f32 + SCALE[step % SCALE.len()];

            let mut osc = ctx.create_oscillator();
            osc.set_type(OscillatorType::Sawtooth);
            osc.frequency().set_value(55. * 2_f32.powf(semitones / 12.));
            vibrato_depth.connect(osc.detune());

            let filter = ctx.create_biquad_filter();
            filter.q().set_value(5.);
            filter.frequency().set_value(2500.);
            cutoff_depth.connect(filter.frequency());

            let amp = ctx.create_gain();
            amp.gain().set_value(0.5);
            amp_depth.connect(amp.gain());

            osc.connect(&filter);
            filter.connect(&amp);
            amp.connect(&bus);
            osc.start();
        }
    }
}
