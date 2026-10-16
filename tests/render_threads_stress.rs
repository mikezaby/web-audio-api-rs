//! Render with 4 threads while another thread changes the graph at random (stage B step 4)
//!
//! Also run it under ThreadSanitizer, see `docs/parallel-render.md`. `STRESS_SEED` picks another
//! random sequence of changes.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, BiquadFilterNode, DelayNode, GainNode, OscillatorNode,
    PannerNode, StereoPannerNode,
};
use web_audio_api::worklet::{
    AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
    AudioWorkletProcessor,
};
use web_audio_api::AudioParam;

const RUN_FOR: Duration = Duration::from_secs(3);
const MAX_NODES: usize = 48;
const GROUPS: u32 = 6;

enum Kind {
    Oscillator(OscillatorNode),
    Gain(GainNode),
    Biquad(BiquadFilterNode),
    Delay(DelayNode),
    StereoPanner(StereoPannerNode),
    Panner(PannerNode),
}

struct Node {
    kind: Kind,
    group: u32,
}

impl Node {
    fn create(ctx: &AudioContext, rng: &mut StdRng) -> Self {
        let group = rng.random_range(0..GROUPS);
        ctx.set_render_group(group);
        let kind = match rng.random_range(0..6) {
            0 => {
                let mut osc = ctx.create_oscillator();
                osc.frequency().set_value(rng.random_range(50.0..2000.0));
                osc.start();
                if rng.random_bool(0.5) {
                    // ends on its own, so it leaves the graph once its handle drops
                    osc.stop_at(ctx.current_time() + rng.random_range(0.0..0.2));
                }
                Kind::Oscillator(osc)
            }
            1 => Kind::Gain(ctx.create_gain()),
            2 => Kind::Biquad(ctx.create_biquad_filter()),
            3 => Kind::Delay(ctx.create_delay(0.1)),
            4 => Kind::StereoPanner(ctx.create_stereo_panner()),
            // merges its group into group 0, through the listener
            _ => Kind::Panner(ctx.create_panner()),
        };
        Self { kind, group }
    }

    fn node(&self) -> &dyn AudioNode {
        match &self.kind {
            Kind::Oscillator(node) => node,
            Kind::Gain(node) => node,
            Kind::Biquad(node) => node,
            Kind::Delay(node) => node,
            Kind::StereoPanner(node) => node,
            Kind::Panner(node) => node,
        }
    }

    fn param(&self) -> &AudioParam {
        match &self.kind {
            Kind::Oscillator(node) => node.frequency(),
            Kind::Gain(node) => node.gain(),
            Kind::Biquad(node) => node.frequency(),
            Kind::Delay(node) => node.delay_time(),
            Kind::StereoPanner(node) => node.pan(),
            Kind::Panner(node) => node.position_x(),
        }
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
        true
    }
}

/// Pick a node to connect `from` into: mostly its own group, else mostly a later group, which
/// keeps groups apart, and sometimes an earlier one, which merges them
fn pick_target<'a>(nodes: &'a [Node], from: &Node, rng: &mut StdRng) -> &'a Node {
    let wanted = |node: &&Node| match rng.random_range(0..10) {
        0..=5 => node.group == from.group,
        6..=8 => node.group > from.group,
        _ => node.group < from.group,
    };
    let candidates: Vec<&Node> = nodes.iter().filter(wanted).collect();
    if candidates.is_empty() {
        &nodes[rng.random_range(0..nodes.len())]
    } else {
        candidates[rng.random_range(0..candidates.len())]
    }
}

#[test]
fn render_threads_survive_random_graph_changes() {
    // a panic anywhere fails the test, also when render code catches it to keep rendering
    let panics = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&panics);
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        counter.fetch_add(1, Ordering::Relaxed);
        default_hook(info);
    }));

    let seed = std::env::var("STRESS_SEED").map_or(1, |seed| seed.parse().unwrap());
    println!("STRESS_SEED={seed}");
    let mut rng = StdRng::seed_from_u64(seed);

    let ctx = AudioContext::new(AudioContextOptions {
        sink_id: "none".into(),
        render_threads: 4,
        ..AudioContextOptions::default()
    });

    // one probe per group, kept alive, to see that groups really render on several threads
    let threads = Arc::new(Mutex::new(HashSet::new()));
    let probes: Vec<_> = (1..GROUPS)
        .map(|group| {
            ctx.set_render_group(group);
            let options = AudioWorkletNodeOptions {
                processor_options: Arc::clone(&threads),
                ..AudioWorkletNodeOptions::default()
            };
            let probe = AudioWorkletNode::new::<ThreadProbe>(&ctx, options);
            probe.connect(&ctx.destination());
            probe
        })
        .collect();

    let mut nodes: Vec<Node> = vec![];
    let mut changes = 0;
    let start = Instant::now();
    while start.elapsed() < RUN_FOR {
        match rng.random_range(0..10) {
            0..=2 if nodes.len() < MAX_NODES => nodes.push(Node::create(&ctx, &mut rng)),
            3..=5 if !nodes.is_empty() => {
                let from = &nodes[rng.random_range(0..nodes.len())];
                match rng.random_range(0..4) {
                    0 => {
                        from.node().connect(&ctx.destination());
                    }
                    1 => {
                        from.node()
                            .connect(pick_target(&nodes, from, &mut rng).param());
                    }
                    _ => {
                        // also into itself and back up the chain, making cycles
                        let to = pick_target(&nodes, from, &mut rng);
                        if to.node().number_of_inputs() > 0 {
                            from.node().connect(to.node());
                        }
                    }
                }
            }
            6 if !nodes.is_empty() => nodes[rng.random_range(0..nodes.len())].node().disconnect(),
            7 | 8 if !nodes.is_empty() => {
                // drop the handle, the render thread frees the node once it ended
                nodes.swap_remove(rng.random_range(0..nodes.len()));
            }
            9 => {
                let oscillator = nodes.iter_mut().find_map(|node| match &mut node.kind {
                    Kind::Oscillator(osc) => Some(osc),
                    _ => None,
                });
                if let Some(osc) = oscillator {
                    osc.stop();
                }
            }
            _ => continue,
        }
        changes += 1;
        if changes % 4 == 0 {
            // let a few quanta render between bursts of changes
            std::thread::sleep(Duration::from_micros(500));
        }
    }
    let rendered_on = threads.lock().unwrap().len();
    println!(
        "{changes} changes, {} nodes left, groups rendered on {rendered_on} threads",
        nodes.len()
    );

    let rendered = ctx.current_time();
    std::thread::sleep(Duration::from_millis(100));
    assert!(ctx.current_time() > rendered, "rendering stopped");

    drop(nodes);
    drop(probes);
    ctx.close_sync();
    assert_eq!(panics.load(Ordering::Relaxed), 0, "a thread panicked");
    assert!(rendered_on > 1, "groups never rendered in parallel");
}
