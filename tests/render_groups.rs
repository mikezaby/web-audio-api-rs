use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, OscillatorType};
use web_audio_api::AudioBuffer;

const LENGTH: usize = 48000 / 2;

/// A graph with every kind of edge between render groups, rendered with or without groups
fn render(grouped: bool) -> AudioBuffer {
    let mut ctx = OfflineAudioContext::new(2, LENGTH, 48000.);
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
    let plain = render(false);
    let grouped = render(true);

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

#[test]
fn grouped_renders_are_bit_identical() {
    let first = render(true);
    let second = render(true);

    for channel in 0..2 {
        assert_eq!(
            first.get_channel_data(channel),
            second.get_channel_data(channel)
        );
    }
}
