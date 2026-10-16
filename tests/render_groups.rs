use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, OscillatorType};
use web_audio_api::worklet::{
    AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
    AudioWorkletProcessor,
};
use web_audio_api::AudioBuffer;

const LENGTH: usize = 48000 / 2;

/// A graph with every kind of edge between render groups, rendered with or without groups
fn render(grouped: bool, threads: usize) -> AudioBuffer {
    let mut ctx = OfflineAudioContext::new(2, LENGTH, 48000.);
    ctx.set_render_threads(threads);
    let group = |group| {
        if grouped {
            ctx.set_render_group(group);
        }
    };

    group(0);
    let master = ctx.create_gain();
    master.gain().set_value(0.5);
    master.connect(&ctx.destination());
    // a group 0 source feeding a track merges that track into group 0
    let mut drone = ctx.create_oscillator();
    drone.frequency().set_value(55.);
    drone.start();

    group(3);
    let mut lfo = ctx.create_oscillator();
    lfo.frequency().set_value(3.);
    let depth = ctx.create_gain();
    depth.gain().set_value(400.);
    lfo.connect(&depth);
    let tremolo = ctx.create_gain();
    tremolo.gain().set_value(0.2);
    lfo.connect(&tremolo);
    lfo.start();

    group(1);
    let mut saw = ctx.create_oscillator();
    saw.set_type(OscillatorType::Sawtooth);
    saw.frequency().set_value(220.);
    let filter = ctx.create_biquad_filter();
    filter.frequency().set_value(800.);
    depth.connect(filter.frequency()); // param input from group 3
    let pan = ctx.create_stereo_panner(); // mono to stereo
    pan.pan().set_value(-0.5);
    let feedback = ctx.create_gain();
    feedback.gain().set_value(0.4);
    let delay = ctx.create_delay(0.5);
    delay.delay_time().set_value(0.1);
    saw.connect(&filter);
    filter.connect(&pan);
    pan.connect(&delay);
    delay.connect(&feedback);
    feedback.connect(&delay);
    pan.connect(&master);
    delay.connect(&master);
    saw.start();

    group(2);
    let mut square = ctx.create_oscillator();
    square.set_type(OscillatorType::Square);
    square.frequency().set_value(330.);
    let panner = ctx.create_panner(); // merges into group 0 through the listener
    panner.position_x().set_value(2.);
    let amp = ctx.create_gain();
    amp.gain().set_value(0.3);
    tremolo.connect(amp.gain()); // a second consumer of group 3
    square.connect(&amp);
    amp.connect(&panner);
    amp.connect(&master); // two groups summing into one input
    panner.connect(&master);
    square.start();

    group(4);
    let track = ctx.create_gain();
    drone.connect(&track);
    track.connect(&master);

    ctx.start_rendering_sync()
}

#[test]
fn grouped_render_matches_ungrouped() {
    let plain = render(false, 1);
    let grouped = render(true, 1);

    for channel in 0..2 {
        let plain = plain.get_channel_data(channel);
        let grouped = grouped.get_channel_data(channel);
        assert!(plain.iter().any(|&s| s.abs() > 0.1), "graph renders sound");
        assert!(
            plain.iter().all(|&s| s.abs() < 2.),
            "graph stays near full scale"
        );
        for (i, (a, b)) in plain.iter().zip(grouped).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "channel {channel} sample {i}: {a} != {b}"
            );
        }
    }
}

/// Short notes in several groups, their handles dropped so they leave the graph when they end
fn render_notes(grouped: bool, threads: usize) -> AudioBuffer {
    let mut ctx = OfflineAudioContext::new(1, LENGTH, 48000.);
    ctx.set_render_threads(threads);
    let master = ctx.create_gain();
    master.gain().set_value(0.2);
    master.connect(&ctx.destination());

    for note in 0..24 {
        if grouped {
            ctx.set_render_group(note % 4 + 1);
        }
        let start = note as f64 * 0.02;
        let mut osc = ctx.create_oscillator();
        osc.frequency().set_value(110. * (1. + note as f32 / 4.));
        let amp = ctx.create_gain();
        amp.gain().set_value(0.5);
        osc.connect(&amp);
        amp.connect(&master);
        osc.start_at(start);
        osc.stop_at(start + 0.05);
    }

    ctx.start_rendering_sync()
}

#[test]
fn grouped_render_with_ending_nodes_matches_ungrouped() {
    let plain = render_notes(false, 1);
    let grouped = render_notes(true, 1);

    let plain = plain.get_channel_data(0);
    let grouped = grouped.get_channel_data(0);
    assert!(plain.iter().any(|&s| s.abs() > 0.1), "graph renders sound");
    for (i, (a, b)) in plain.iter().zip(grouped).enumerate() {
        assert!((a - b).abs() <= 1e-6, "sample {i}: {a} != {b}");
    }
}

#[test]
fn grouped_renders_are_bit_identical_for_any_thread_count() {
    let one = render(true, 1);
    let notes_one = render_notes(true, 1);

    for threads in [1, 2, 4] {
        let many = render(true, threads);
        for channel in 0..2 {
            assert_eq!(
                one.get_channel_data(channel),
                many.get_channel_data(channel),
                "{threads} threads"
            );
        }
        let notes = render_notes(true, threads);
        assert_eq!(
            notes_one.get_channel_data(0),
            notes.get_channel_data(0),
            "{threads} threads, nodes ending"
        );
    }
}

/// Records which threads it renders on
struct ThreadProbe(Arc<Mutex<HashSet<ThreadId>>>);

impl AudioWorkletProcessor for ThreadProbe {
    type ProcessorOptions = Arc<Mutex<HashSet<ThreadId>>>;

    fn constructor(threads: Self::ProcessorOptions) -> Self {
        Self(threads)
    }

    fn process<'a, 'b>(
        &mut self,
        _inputs: &'b [&'a [&'a [f32]]],
        _outputs: &'b mut [&'a mut [&'a mut [f32]]],
        _params: AudioParamValues<'b>,
        _scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        self.0.lock().unwrap().insert(std::thread::current().id());
        // long enough for helper threads to take groups before the render thread gets to them
        std::thread::sleep(Duration::from_micros(200));
        true
    }
}

/// Number of threads that rendered 8 groups for 20 quanta
fn threads_used(render_threads: usize) -> usize {
    let mut ctx = OfflineAudioContext::new(1, 128 * 20, 48000.);
    ctx.set_render_threads(render_threads);
    let threads = Arc::new(Mutex::new(HashSet::new()));
    for group in 1..=8 {
        ctx.set_render_group(group);
        let options = AudioWorkletNodeOptions {
            processor_options: Arc::clone(&threads),
            ..AudioWorkletNodeOptions::default()
        };
        let probe = AudioWorkletNode::new::<ThreadProbe>(&ctx, options);
        probe.connect(&ctx.destination());
    }
    let _ = ctx.start_rendering_sync();
    let count = threads.lock().unwrap().len();
    count
}

#[test]
fn groups_render_on_several_threads() {
    assert_eq!(threads_used(1), 1);
    assert!(threads_used(4) > 1);
}
