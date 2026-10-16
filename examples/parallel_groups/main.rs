use std::time::{Duration, Instant};

use web_audio_api::context::{AudioContext, AudioContextLatencyCategory, AudioContextOptions};
use web_audio_api::AudioRenderCapacityOptions;

mod tracks;

// Play the `parallel_groups` bench graph in real time and print the render load every second.
// A clock speed (context time over wall time) below 1.0 means rendering cannot keep up.
//
// `cargo run --release --example parallel_groups [voices per track, default 24]`
// `RENDER_GROUPS=1 cargo run --release --example parallel_groups` puts each track in its own
// render group, `RENDER_THREADS=4` also renders the groups on 4 threads.
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example parallel_groups`
fn main() {
    env_logger::init();

    let voices = std::env::args().nth(1).map_or(tracks::VOICES, |arg| {
        arg.parse().expect("voices per track must be a number")
    });

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };
    let render_threads = std::env::var("RENDER_THREADS").map_or(1, |threads| {
        threads.parse().expect("RENDER_THREADS must be a number")
    });
    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        render_threads,
        ..AudioContextOptions::default()
    });

    // threads only help with groups to spread over them
    let grouped = std::env::var("RENDER_GROUPS").is_ok() || render_threads > 1;
    tracks::build(&context, voices, grouped);
    println!(
        "{} tracks x {voices} voices, render groups {grouped}, render threads {render_threads}, \
         Ctrl-C to stop",
        tracks::TRACKS
    );

    let mut previous: Option<(f64, Instant)> = None;
    let capacity = context.render_capacity();
    capacity.set_onupdate(move |e| {
        let now = Instant::now();
        if let Some((timestamp, wall)) = previous {
            let clock = (e.timestamp - timestamp) / now.duration_since(wall).as_secs_f64();
            println!(
                "clock {clock:.3}  load avg {:3.0}% peak {:3.0}%  underruns {:5.1}%",
                e.average_load * 100.,
                e.peak_load * 100.,
                e.underrun_ratio * 100.
            );
        }
        previous = Some((e.timestamp, now));
    });
    capacity.start(AudioRenderCapacityOptions {
        update_interval: 1.,
    });

    loop {
        std::thread::sleep(Duration::from_secs(60));
    }
}
