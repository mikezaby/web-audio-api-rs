# Parallel render groups

Status: stage A steps 1 to 5 done on `perf/parallel-render`, stage B steps 1 to 3 working (spin tuning and the stress test remain); the stage A exit checks through `node-web-audio-api` remain.
Branch: `perf/parallel-render` in this clone (`origin` is upstream `orottier/web-audio-api-rs`, never push there).
Written 2026-10-15 as a handover from the Blibliki repository (`~/projects/blibliki/blibliki`).

## Why

Blibliki is a web audio instrument that also plays on a Raspberry Pi 5 through `node-web-audio-api`, which wraps this crate.
An `AudioContext` here renders its whole graph on one thread, node after node (`Graph::render`, `src/render/graph.rs:495`).
A full 8-track Blibliki instrument therefore uses one of the Pi's four cores.

The goal is to render independent parts of one graph on several cores inside the same render quantum, the way Ardour renders its routes.
The same quantum means no added latency.

## Goals and non-goals

Goals:

- Opt-in twice: nodes must be tagged with a render group, and the context must be given more than one render thread.
  Without both, behaviour and output stay exactly as upstream.
- No added latency: every group renders inside the quantum it belongs to.
- Works for `AudioContext` and `OfflineAudioContext`.
  Offline renders get faster too, which Blibliki uses for recipe renders and tests.

Non-goals:

- Automatic partitioning of the graph.
  It is the way to make this acceptable upstream, and it can replace the hint later without touching the scheduler.
- Splitting one chain of nodes across cores.
  Ardour and Ableton do not do it either: one chain is serial by nature.
- Browsers. They give each context one real-time render thread and nothing here changes that.

## Gates before any code here

These run in Blibliki, on the Pi, and each one can end the project.

1. Measure the heavy recipes (Showcase, dnb, ambient, dark-ambient) on the Pi with the optimised `node-web-audio-api` (fork main `ae48481`) plus the idle-worker fix (branch `perf/idle-worker-sleep`).
   Measure clock speed (delta of `context.currentTime` over delta of wall time), not `renderCapacity`, which misled before.
   If everything plays in real time, stop.
2. Apply the worklet cuts in Blibliki `docs/findings.md`, section "Showcase does not play cleanly on the Pi", and measure again.
3. Find the ceiling.
   Profile one heavy recipe and measure the share of the heaviest track and the share of the serial tail (mixer, shared reverb send, master).
   The best possible speedup is about `1 / (heaviest track + tail)`, capped by the core count.
   If that is below 1.5x, stop.
4. Two-process spike on the Pi: tracks 1 to 4 in one Node process, tracks 5 to 7 in another, each a PipeWire client to the playback device.
   Measure clock speed of each.
   This confirms the parallel gain on the real machine before weeks of work here.

## The pattern to copy: Ardour's process graph

Ardour schedules routes (tracks and buses), not plugins.
Read `libs/ardour/graph.cc` at <https://github.com/Ardour/ardour/blob/master/libs/ardour/graph.cc> and its neighbours in `libs/ardour/`.

- Each node counts how many nodes feed it.
  Nodes with no inputs go on a trigger queue at the start of each cycle.
- A pool of DSP threads, one per core, takes nodes off the queue.
  The audio callback thread is one of them.
- A finished node decrements the count of every node it feeds.
  A node that reaches zero goes on the queue.
- Idle threads sleep on a semaphore.
  A thread that finds work wakes at most as many idle threads as there is queued work.
- The callback thread waits once per cycle, until the last terminal node finishes.

Licence: Ardour is GPL-2.0-or-later and this crate is MIT.
Read Ardour for the design only.
Do not copy its code here.

## How the crate renders today

The plan depends on these facts, checked on `main` at `03d0cb2`.

- Rendering runs inside the audio backend callback.
  `RenderThread::render_inner` (`src/render/thread.rs:441`) renders 128-frame quanta in a loop, calling `Graph::render` once per quantum (`:503`).
  Offline rendering goes through `render_offline_quantum` (`:355`).
  Control messages are handled on this thread between quanta (`:475`, `:530`).
- `Graph::render` walks `ordered`, the topological order, on one thread (`src/render/graph.rs:495`).
- Push model: after a node processes, it adds each output straight into the input buffer of the nodes it feeds (the edge loop at `graph.rs:528`).
- `AudioRenderQuantumChannel::add` reuses the producer's buffer when the input is silent: `*self = other.clone()` (`src/render/quantum.rs:114`).
  Sharing a buffer between two groups must therefore never happen through this path.
- Buffers are `Rc` handles from one per-graph pool (`quantum.rs:13` to `:19`), so they are neither `Send` nor `Sync`.
- Nodes live in `RefCell`s (`src/render/node_collection.rs:29`).
- Processors read their `AudioParam`s through the whole node collection (`AudioParamValues`, `src/render/processor.rs:206`).
  `PannerNode` also reads the nine `AudioListener` params this way (`:231`).
- `AudioWorkletGlobalScope` holds the current node id in a `Cell` (`processor.rs:25`) and is built once per quantum (`thread.rs:494`).
- The order is rebuilt lazily when `ordered` is cleared (`add_edge`, `graph.rs:270`).
  `order_nodes` (`:423`) and `visit` (`:336`) break cycles at `DelayNode`s, whose writer is marked as cycle breaker (`src/context/concrete_base.rs:351`, `src/node/delay.rs:364`).
- `DelayNode`'s writer and reader share a ring buffer through `Rc<RefCell<..>>` (`delay.rs:304`, `:376`, `:491`).
  No other node shares `Rc` state between processors (the `Rc` in `src/worklet.rs:707` is test only).
- Nodes are removed during the render loop, and removal rewrites the edges of every other node (`graph.rs:557` to `:590`).
- Nodes are registered through `ConcreteBaseAudioContext::register` (`concrete_base.rs:232`), which sends `ControlMessage::RegisterNode` (`src/message.rs:15`, handled at `thread.rs:140`).
- The crate does not raise the priority of any thread it spawns.
  The callback thread gets its priority from the backend.

## Design

### API, non-standard and opt-in

- `BaseAudioContext::set_render_group(&self, group: u32)`.
  Nodes registered after the call belong to that group.
  Group 0 is the main group and the default.
  Store the current group in `ConcreteBaseAudioContext` and add a `render_group: u32` field to `ControlMessage::RegisterNode`.
  A constructor registers its node, its `AudioParam`s and, for `DelayNode`, both writer and reader in one go, so they all land in the same group with no extra code.
- A node's group is fixed at registration.
- `AudioContextOptions::render_threads: usize`, default 1.
  For offline contexts, `OfflineAudioContext::set_render_threads(n)`, valid before rendering starts.
- `AudioWorkletGlobalScope::render_group()`, so a processor knows which group it renders in.
  Stage C needs it.

### Group rules

- The destination, the `AudioListener` and its params stay in group 0.
- Groups joined by a hidden edge (`other_index == usize::MAX`) are merged, because the receiver reads the sender's buffer straight from the node collection.
  This keeps `AudioParam`s with their owner, and merges any group with a `PannerNode` into group 0, because the listener reaches each panner through a hidden edge.
  Blibliki only uses `StereoPannerNode`, so this costs Blibliki nothing.
- Groups that depend on each other in both directions, directly or through other groups, are merged into one.
  These are the strongly connected components of the group graph (Tarjan).
  Without the merge, the dependency counts deadlock.
  Edges cleared by a cycle breaker do not count.
- Merging is recomputed every time `ordered` is rebuilt.

### Rendering with groups

- At the end of `order_nodes`, build the group plan:
  - the node to group map after merging;
  - `ordered` filtered per group (a filtered topological order stays valid inside the group);
  - each group's initial dependency count and dependents;
  - the root groups;
  - the list of edges that cross groups.
- Per-group state: its ordered node list, its incoming mailboxes and a list of nodes to drop.
- Buffer pools are per render group id (the tag), not per merged group, and every edge between two render groups copies: through a mailbox between merged groups, straight into the other pool within a merged group.
  So no `Rc` ever leaves its render group, however groups merge now or split later.
  Pools per merged group could not promise that, because buffers held inside processors (the `DelayNode` ring buffer) cannot move to another pool when a merge splits.
- The node collection stays one, indexed by node id.
  The hidden-edge rule already keeps every node a processor touches (its own, its params, the listener params for a panner) in its merged group, so a thread only ever touches the nodes of its own group.
  Lookups by id (`route_message`, the channel setters, `diagnostics`, the lifecycle code) stay as they are.
- Edges between groups go through mailboxes.
  A mailbox is preallocated plain memory (a `Vec<f32>` per channel, a channel count and a silent flag), with no `Rc` inside.
  The producer copies its output into the mailbox after processing.
  The consumer, before its group renders, adds each incoming mailbox into the target input through its own pool, in a fixed order (producer node id, then output index).
  The fixed order keeps sums deterministic for any thread count.
  An `AudioParam` input from another group is a mailbox like any other.
- Node removal is recorded in the group's drop list and applied on the callback thread after all groups finish, using the existing removal code, then the order is invalidated.
- With no groups other than group 0, the current single loop runs unchanged.

### Worker pool, Ardour style

- `render_threads - 1` helper threads, spawned when the context starts.
  The callback thread works as well.
- Each quantum: reset every group's pending count (`AtomicU32`) from its initial count, put the root groups on the trigger queue, and wake at most as many idle helpers as there are queued groups beyond the first.
- Worker loop: take a group, build its scope (same frame and time, `render_group` set), render it, fill its outgoing mailboxes, then decrement each dependent with `AcqRel`.
  A dependent that reaches zero goes on the queue and wakes one idle helper.
  The last group to finish decrements `remaining_groups` to zero and releases the callback thread.
- Idle workers spin for a short, configurable time, then park (`crossbeam-utils` `Parker`).
  Never spin without limit: on the Pi, a spinning worker takes a whole core.
- Trigger queue: a bounded `crossbeam-channel` (already a dependency) used as a queue of group ids, preallocated, with no allocation per quantum.
- Real-time priority for helpers on online contexts through the `audio_thread_priority` crate.
  If the system refuses (Linux without rtkit or `RLIMIT_RTPRIO`), carry on at normal priority and log once.
  Offline contexts use normal priority.
- Soundness: the node collection, the group state and the mailboxes hold `Rc` and `RefCell`, so share them through a wrapper with `unsafe impl Sync` and a `SAFETY:` comment stating the invariant.
  One thread touches a group's nodes, pools and drop list between its trigger and its finish.
  A group is triggered only after all its producers finish, ordered by `AcqRel` on the counters.
  No `Rc` leaves its render group (pools per render group, copies on every edge between render groups, a `DelayNode`'s pair in one render group).
  The node collection does not change while groups render (removal waits for the end of the quantum, registration happens between quanta).
- The merged group holding the destination renders on the callback thread, because the render thread keeps a clone of the destination output between callbacks (`buffer_offset` in `src/render/thread.rs`).
- Panics: keep the per-node `catch_unwind`.
  A worker must still decrement its counters after a panic (use a drop guard) so the quantum completes.
- Control messages and order rebuilds stay on the callback thread between quanta, as today.

## Stages

Commit each numbered step with its tests.
Measure before committing any step that claims a speedup.

### Stage A: groups on one thread (about 1 to 2 weeks)

1. `set_render_group`, the `RegisterNode` field and the node to group map.
   Unit tests: params and the `DelayNode` pair land in their owner's group.
2. The group plan in `order_nodes`: merge rules, per-group order, counts, roots.
   Unit tests in `src/render/graph/test.rs`: two-way dependency merge, three-group cycle merge, panner merge, filtered order stays topological.
3. Mailboxes for edges between groups, and rendering groups one after another in dependency order.
   Registering or dropping a node while groups are in use rebuilds the whole plan.
4. Buffer pools per render group, and copies on edges between render groups within a merged group.
   Built differently from the first draft of this plan (per-group node collections), see "Rendering with groups".
5. Deferred node removal.

Exit:

- `cargo test` and `cargo clippy --all-targets` pass, and existing tests are unchanged.
- New offline test: the same graph rendered with and without groups differs by at most 1e-6 per sample (summing order changes), and two grouped renders are bit-identical.
- Criterion: the path without groups is unchanged, and the grouped path on one thread is at most 5% slower.
- Linked into `node-web-audio-api` through the path dependency (its `Cargo.toml:18` has the line commented out), the WPT suite fails exactly the tests it fails on base.

Measured after step 3, M1 Pro, one thread, `cargo bench --bench parallel_groups`: 3.79 s without groups, 3.77 s with 8 groups, against 3.73 s on base.
The small graphs in `my_benchmark` stay within 1% of base, and `bench_graph_reordering` within 2%.

Measured after step 5, same machine: 3.58 s without groups, 3.60 s with 8 groups.
Small graphs read up to 3% slower than base in plain builds, but within 1% with every function and branch target aligned (`RUSTFLAGS="-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=5"` on both sides), so the rest is code layout.
On Linux, the `iai` instruction counts (`cargo bench --no-default-features --features iai`) settle this without layout noise.

### Stage B: worker pool (about 1 to 2 weeks)

1. `render_threads` options and helper thread lifecycle (spawn on start, join on close, and no threads at all when the count is 1).
2. Scheduler: counters, trigger queue, spin then park, completion.
3. Real-time priority.
4. Stress test and benchmark.

Exit:

- Grouped renders are bit-identical for 1, 2 and 4 render threads.
- Stress test: an online context with sink `"none"` (`src/context/online.rs:126`) renders with 4 threads while another thread randomly creates, connects, disconnects and drops nodes across groups.
  Run it under ThreadSanitizer (nightly, `-Zsanitizer=thread`).
- `benches/parallel_groups.rs`, on the graph in `examples/parallel_groups/tracks.rs`: 8 groups of 24 voices (sawtooth, biquad, gain, all modulated by a track LFO) into a bus with a `ConvolverNode` send, summed into a gain to the destination, 10 seconds offline.
  Base, one thread: 3.73 s on an M1 Pro.
  At least 3x at 4 threads on an M1.
  Record the Pi number next to it.
  `cargo run --release --example parallel_groups` plays the same graph in real time and prints clock speed, load and underruns.
  A small graph with one group is at most 2% slower than upstream.
- On the Pi, online at PipeWire quantum 1024, no underruns.

Measured with the prototype (steps 1 and 2), M1 Pro, `cargo bench --bench parallel_groups`: 3.59 s without groups, 3.68 s with 8 groups on one thread, 1.94 s on 2 threads (1.85x), 1.16 s on 4 threads (3.10x).
In real time at 64 voices per track, 4 threads halve the average load (45% against 89%) but underruns stay (5% of callbacks against 3 to 7% before), with peaks up to 350% of the callback budget.
Helpers at normal priority get preempted while the callback thread waits for their group, so step 3 comes next.

With step 3 (real-time priority, `audio_thread_priority` without D-Bus), back to back at 64 voices per track: 1 thread misses 18.2% of callbacks at 92% load; 4 threads miss 0.6% at 35% load, peaks up to 122%, in 10 of 18 seconds.
The remaining misses are single callbacks.
Spinning before a helper sleeps makes them worse, because at real-time priority it counts against the time-constraint budget: 0.2% missed callbacks without spin, 1.1% with a 50 us spin, 34% with a 3 ms spin (macOS, 4 threads, 64 voices per track).
So helpers sleep as soon as the queue is empty, and the "spin, then park" item above is settled as "park".

### Stage C: one JS worklet worker per group, in `node-web-audio-api`

Without it, every worklet call from every group queues on one JS worker, and Blibliki's sound is mostly worklets.
Work in the `node-web-audio-api` fork (`~/projects/blibliki/node-web-audio-api`, main `ae48481`).

1. One process-call channel and one JS `Worker` per context and group, created lazily.
   Today there is one per context: `allocate_process_call_channel` (`src/audio_worklet_node.rs:77`), the node picks its sender at construction (`:783` to `:796`), and the `Worker` starts in `js/AudioWorklet.js:137`.
   `addModule` replays every module already loaded into each new worker.
2. JS API: `context.setRenderGroup(n)` and `renderThreads` in the `AudioContext` options.
3. Mark the deviation with a `ponytail:` comment: with groups, there is one `AudioWorkletGlobalScope` per group instead of per context.
   Processors that share module-level state across nodes would stop sharing it.
   Blibliki's processors do not (checked 2026-10-15: the wavetable's `__blibliki_wt` cache is keyed by each processor's own state).
4. Check the polling from `ae48481` (render side 30 us, worker side 50 us) with several render thread and worker pairs on four cores.

Exit: WPT AudioWorklet results match base, and the Blibliki recipes that failed gate 1 reach clock speed 1.0 on the Pi.

### Stage D: Blibliki uses it

Feature-detect `setRenderGroup`.
The instrument sets each track's group while it builds that track's modules, including live mod matrix additions through `InstrumentSession.applyEngineChanges`.
The master stays in group 0.
Blibliki records the dependency on a non-standard API in an ADR when this lands.
Exit: Blibliki `pnpm tsc`, `pnpm lint`, `pnpm test`, and a device check on the Pi with the XL3.

### Stage E, optional: voice banks

A Blibliki track is 8 voice chains (source, filter, amp, each 8 voices by default) summed into one effects chain with a single voice.
Up to that sum the voices do not depend on each other, so a heavy track can be split into banks of voices, each its own render group.

- It only helps when one track is longer than its share: the heaviest track takes more than the total track work divided by the core count.
  Gate 3 measures exactly this.
  With balanced tracks it adds groups, mailbox copies and syncs for no gain.
- It conflicts with Blibliki worklet cut 2, which renders every voice of a poly envelope in one worklet.
  Do that cut first, because it lowers total work on any core count.
  A bank then gets its own poly worklet, so split into 2 or 4 banks, not 8 single voices.
- Blibliki side: the engine's `PolyModule` sets the bank's group while it builds each voice, and the track's LFOs stay in a group before the banks.

## Working notes

- `origin` here and in the `node-web-audio-api` clone is upstream.
  Never push to it.
  Sharing goes through a fork remote, which is the user's decision.
- Build `node-web-audio-api` on macOS 27 with `DEVELOPER_DIR=/Library/Developer/CommandLineTools npm run build:only`.
- Upstream is unlikely to take a non-standard hint.
  To offer this upstream later, replace the hint with automatic partitioning (for example, the subgraphs that feed separate inputs of a summing node) and keep the scheduler.

## Risks and open questions

- Sync cost per quantum.
  At PipeWire quantum 1024, each callback renders 8 quanta back to back, and each quantum syncs the workers once.
  Expect a few microseconds per sync and measure it on the Pi.
- Each group with worklets pairs a render worker with a JS worker.
  Four cores with four pairs may contend.
  Measure before raising the thread count past the core count.
- Real-time priority on the Pi depends on rtkit and PipeWire limits.
- The shared node collection interleaves nodes of different groups, so threads may write borrow flags on the same cache line at group boundaries.
  Nodes are hundreds of bytes, so this should be rare; revisit if it shows up in stage B profiles.
