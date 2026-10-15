# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Pure Rust implementation of the W3C Web Audio API (crate `web-audio-api`) for non-browser use. MSRV is 1.88 (`rust-version` in `Cargo.toml`, also pinned in the CI `msrv` job).

## Commands

```sh
cargo build --all-targets
cargo test --all-targets
cargo test --lib render::graph::tests::test_cycle       # single unit test (module path filter)
cargo test --test offline test_offline_render           # single integration test in tests/offline.rs
cargo run --release --example biquad                    # examples/ use audio files from samples/
```

What CI and pre-commit enforce (match these before considering work done):

```sh
cargo fmt --all
cargo clippy --locked --workspace --all-targets -- -D warnings
RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --no-deps --document-private-items   # broken intra-doc links fail
pre-commit run --all-files    # also markdownlint, prettier (md/yaml), codespell (.codespellignore)
```

- CI runs on Linux with `--all-features` (needs ALSA/JACK/PipeWire dev libs). On macOS, `--all-features` pulls in `cpal-jack`/`cpal-pipewire`, so stick to default features locally. The `cubeb` feature (used by the pre-commit check/doc hooks) needs `cmake`.
- `Cargo.lock` is gitignored (library crate); CI generates it.
- Benchmarks: `cargo bench` (criterion), or `cargo bench --no-default-features --features iai` (valgrind instruction counts; this is what a `/bench` comment on a PR runs, comparing against `main`).
- User-facing changes get a line under `## Unreleased` in `CHANGELOG.md`. The release workflow extracts the `## Version X.Y.Z (date)` section as release notes.

## Architecture

### Control thread vs render thread

Everything is split between the **control thread** (user-facing objects: contexts, `AudioNode`s, `AudioParam`s) and the **render thread** (`render::RenderThread` driving a `render::graph::Graph` of `AudioProcessor`s). They share no mutable state; the only channels are:

- control → render: `ControlMessage` (`src/message.rs`) over a crossbeam channel, sent via `ConcreteBaseAudioContext::send_control_msg`. While the context is suspended, messages are buffered on the control side.
- render → control: `EventDispatch` (`src/events.rs`), drained by an `EventLoop` thread that calls user event handlers (`onended`, `onprocessorerror`, worklet messages, ...). The render thread uses `try_send` and must never block on a full channel (`tests/stall_render.rs`).

### Render thread is real-time: no allocations, locks, or blocking deallocations

- Payloads are allocated on the control thread (wrapped in `llq::Node` before sending). `AudioProcessor::onmessage(&mut dyn Any)` should take ownership via `downcast_mut` + `std::mem::replace`/swap rather than letting a heap value drop in the render thread (see the tombstone trick in `AudioParamProcessor::onmessage`, `src/param.rs`).
- Dropped processors and buffers are shipped to a garbage-collector sidecar thread (`spawn_garbage_collector_thread` in `src/render/thread.rs`).
- `alloc_counter::deny_alloc` is used in tests to assert allocation-free paths (`src/render/quantum.rs`).
- A panicking `AudioProcessor` is caught, the node is removed from the graph, and an `ErrorEvent` is sent (`tests/processor_error.rs`).

### Node = control handle + renderer

Each node type in `src/node/` is a pair: a `FooNode` struct implementing `AudioNode` (holds an `AudioContextRegistration`, a `ChannelConfig`, and its `AudioParam`s) and a private `FooRenderer` implementing `AudioProcessor`. Both are built inside `context.base().register(|registration| (node, Box::new(renderer)))`, which allocates the node id and sends `ControlMessage::RegisterNode`. `src/node/gain.rs` is the minimal template. Control-side setters push updates to the renderer with `registration.post_message(value)` → `AudioProcessor::onmessage`.

`FooOptions` structs mirror the spec's WebIDL dictionaries (IDL quoted in a comment above them) and their `Default` must equal the spec defaults.

### AudioParams are graph nodes

`context.create_audio_param(descriptor, &registration)` registers a separate `AudioParamProcessor` node and connects its output into the owning node. The renderer stores the returned `AudioParamId` and reads values with `params.get(&id)` in `process`. Param buffers are either 128 values (a-rate, changing) or **a single value** (constant or k-rate) — renderers must handle both lengths. Audio connected _to_ a param (`node.connect(param)`) is summed in by the param processor.

### Render quantum and graph

- Audio is processed in blocks of `RENDER_QUANTUM_SIZE = 128` frames (`AudioRenderQuantum`, `src/render/quantum.rs`). Channels are `Rc`-shared, copy-on-write, with silent/single-valued fast paths; because of the `Rc`s, `Graph`/`RenderThread` are `!Send` and have hand-written `unsafe impl Send`.
- `Graph::order_nodes` does a topological sort, recomputed only when the topology changes. Cycles are broken only at a `DelayNode` (internally a `DelayWriter`/`DelayReader` pair; the writer is marked as cycle breaker); nodes in a cycle without one are muted.
- Node lifetime (`Node::can_free` in `src/render/graph.rs`): a renderer is dropped only after the control handle is dropped (`ControlHandleDropped`), combined with the bool returned from `process` (tail time: `false` = only transforms inputs, `true` = still producing output) and its connections. Freed `AudioNodeId`s are recycled back to the control thread via an `llq` queue.
- Reserved ids: `0` destination, `1` `AudioListener`, `2..=10` listener params. The listener is registered lazily (its messages are queued until first use).

### Contexts and backends

- `BaseAudioContext` (trait, `src/context/base.rs`) provides the `create_*` factories via `base() -> &ConcreteBaseAudioContext`, which holds the shared implementation. Implementors: `AudioContext` (real-time, `src/context/online.rs`) and `OfflineAudioContext` (renders on the calling thread: `start_rendering_sync`, async `start_rendering`, `suspend`/`suspend_sync` at given times).
- Audio I/O backends live in `src/io/`: `cpal` (default feature), `cubeb` (feature), and `none`. `AudioContextOptions { sink_id: "none".into(), .. }` uses the `NoneBackend` (no hardware), which is how tests exercise `AudioContext`. Prefer `OfflineAudioContext` for deterministic DSP tests.
- `src/worklet.rs` exposes `AudioWorkletNode` + `AudioWorkletProcessor`, the public API for user-defined processors (`examples/worklet*.rs`). The `spec-compliant-worklet-inputs` feature changes inactive-input behavior. The `diagnostics` feature adds structured context/graph reports (`src/context/diagnostics.rs`).
- Decoding (`src/decoding.rs`) uses Symphonia for all formats. The legacy `mp3`/`ogg`/`flac`/... features are no-ops kept for compatibility.

## Conventions

- Spec deviations are intentional and consistent: snake_case names, getter/setter methods instead of attributes, traits for inheritance (`AudioNode`, `AudioScheduledSourceNode`, `BaseAudioContext`).
- Spec exceptions are panics whose message starts with the DOMException name, e.g. `"NotSupportedError - Invalid sample rate ..."`, and every panicking public fn needs a `# Panics` doc section (`clippy::missing_panics_doc`). Use `assert_valid_*` helpers from `src/lib.rs` where they fit.
- `#![deny(missing_debug_implementations)]`: every public type needs a `Debug` impl.
- Public node docs link to MDN and the spec, include a usage snippet (`no_run` when it needs audio hardware), and list relevant `cargo run --release --example ...` commands.
