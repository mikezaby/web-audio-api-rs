//! Non-uniform partitioned convolution, the engine of `ConvolverNode`
//!
//! The impulse response is cut into stages of growing partition size. The
//! head has 128-frame partitions and renders every quantum with no latency.
//! A tail stage with partitions of `B` frames covers the impulse from `2B`
//! on, so its result for one input block is due one whole block after that
//! block ends: the forward FFT, the multiply-accumulate over its partitions
//! and the inverse FFT are spread over that block instead of landing in one
//! quantum.
//!
//! Each stage keeps one frequency-domain delay line per input lane and one
//! accumulator per output, so an output that sums several (lane, impulse
//! channel) terms needs one inverse FFT. Spectra of silent input are flagged
//! instead of computed, and skipped in the multiply-accumulate.

use std::sync::Arc;

use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::RENDER_QUANTUM_SIZE;

/// Partition size of the head stage
const HEAD_BLOCK: usize = RENDER_QUANTUM_SIZE;
/// Partition sizes of the tail stages, each starting at twice its size
const TAIL_BLOCKS: [usize; 2] = [1024, 8192];

/// Number of input lanes: input channel 0 and 1
pub(crate) const LANES: usize = 2;
/// Number of outputs a convolver can render
const OUTPUTS: usize = 2;

/// The (lane, impulse channel) pairs summed into one output
pub(crate) type Terms = &'static [(usize, usize)];

pub(crate) struct Convolver {
    stages: Vec<Stage>,
}

impl Convolver {
    /// Prepare the convolution of the given impulse response channels, which
    /// must all have the same length. This allocates; call it off the render
    /// thread.
    pub(crate) fn new(impulse: &[Vec<f32>]) -> Self {
        let length = impulse[0].len();
        let mut planner = RealFftPlanner::<f32>::new();
        let segment = |start: usize, end: usize| -> Vec<&[f32]> {
            impulse.iter().map(|channel| &channel[start..end]).collect()
        };

        let head_end = (2 * TAIL_BLOCKS[0]).min(length);
        let mut stages = vec![Stage::new(&mut planner, HEAD_BLOCK, &segment(0, head_end))];

        for (index, &block) in TAIL_BLOCKS.iter().enumerate() {
            let start = 2 * block;
            if length <= start {
                break;
            }
            let end = TAIL_BLOCKS
                .get(index + 1)
                .map_or(length, |next| (2 * next).min(length));
            stages.push(Stage::new(&mut planner, block, &segment(start, end)));
        }

        Self { stages }
    }

    /// Render one quantum: `lanes[i]` is the input of lane `i`, `None` when
    /// it is silent, and `outputs[o]` receives the sum of `terms[o]`.
    ///
    /// Returns false when every output is exact silence.
    pub(crate) fn process(
        &mut self,
        lanes: [Option<&[f32]>; LANES],
        terms: &'static [Terms],
        outputs: &mut [&mut [f32]],
    ) -> bool {
        let (head, tails) = self.stages.split_first_mut().unwrap();
        let mut heard = head.process_head(lanes, terms, outputs);
        for stage in tails {
            heard |= stage.process_tail(lanes, terms, outputs);
        }
        heard
    }

    /// True when no input is left to ring out: further silent input renders
    /// exact silence.
    pub(crate) fn is_silent(&self) -> bool {
        self.stages.iter().all(Stage::is_silent)
    }
}

struct Stage {
    /// partition size
    block: usize,
    /// render quanta per block
    quanta: usize,
    partitions: usize,
    /// quantum within the current block, tail stages only
    quantum: usize,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    /// impulse spectra per channel, `partitions` runs of `block + 1` bins,
    /// scaled by the inverse FFT's `1 / 2B`
    impulse: Vec<Vec<Complex<f32>>>,
    lanes: [Lane; LANES],
    outputs: [Output; OUTPUTS],
    /// terms latched for the block being computed, tail stages only
    terms: &'static [Terms],
    time: Vec<f32>,
    scratch: Vec<Complex<f32>>,
}

struct Lane {
    /// the previous and the current block of input
    window: Vec<f32>,
    window_heard: [bool; 2],
    /// the block being received, tail stages only
    incoming: Vec<f32>,
    incoming_heard: bool,
    /// ring of `partitions` input spectra, the newest at `newest`
    spectra: Vec<Complex<f32>>,
    spectra_heard: Vec<bool>,
    newest: usize,
}

struct Output {
    accumulator: Vec<Complex<f32>>,
    accumulator_heard: bool,
    /// result played during the current block, tail stages only
    ready: Vec<f32>,
    ready_heard: bool,
    /// result computed for the next block, tail stages only
    next: Vec<f32>,
    next_heard: bool,
}

impl Stage {
    fn new(planner: &mut RealFftPlanner<f32>, block: usize, impulse: &[&[f32]]) -> Self {
        let fft_length = 2 * block;
        let bins = block + 1;
        let forward = planner.plan_fft_forward(fft_length);
        let inverse = planner.plan_fft_inverse(fft_length);
        let partitions = impulse[0].len().div_ceil(block).max(1);

        let mut time = vec![0.; fft_length];
        let scratch_length = forward.get_scratch_len().max(inverse.get_scratch_len());
        let mut scratch = vec![Complex::default(); scratch_length];

        let scale = 1. / fft_length as f32;
        let impulse = impulse
            .iter()
            .map(|channel| {
                let mut spectra = vec![Complex::default(); partitions * bins];
                for (part, spectrum) in channel.chunks(block).zip(spectra.chunks_mut(bins)) {
                    time.fill(0.);
                    time.iter_mut().zip(part).for_each(|(t, v)| *t = v * scale);
                    forward
                        .process_with_scratch(&mut time, spectrum, &mut scratch)
                        .unwrap();
                }
                spectra
            })
            .collect();

        let lane = || Lane {
            window: vec![0.; fft_length],
            window_heard: [false; 2],
            incoming: vec![0.; block],
            incoming_heard: false,
            spectra: vec![Complex::default(); partitions * bins],
            spectra_heard: vec![false; partitions],
            newest: 0,
        };
        let output = || Output {
            accumulator: vec![Complex::default(); bins],
            accumulator_heard: false,
            ready: vec![0.; block],
            ready_heard: false,
            next: vec![0.; block],
            next_heard: false,
        };

        Self {
            block,
            quanta: block / RENDER_QUANTUM_SIZE,
            partitions,
            quantum: 0,
            forward,
            inverse,
            impulse,
            lanes: [lane(), lane()],
            outputs: [output(), output()],
            terms: &[],
            time,
            scratch,
        }
    }

    fn bins(&self) -> usize {
        self.block + 1
    }

    /// The whole impulse segment, rendered in the quantum it is heard
    fn process_head(
        &mut self,
        lanes: [Option<&[f32]>; LANES],
        terms: &[Terms],
        outputs: &mut [&mut [f32]],
    ) -> bool {
        let block = self.block;

        for (lane, input) in self.lanes.iter_mut().zip(lanes) {
            lane.window.copy_within(block.., 0);
            let heard = receive(&mut lane.window[block..], input);
            lane.window_heard = [lane.window_heard[1], heard];
        }
        self.push_spectra();

        let mut heard = false;
        for (index, (output, &terms)) in outputs.iter_mut().zip(terms).enumerate() {
            self.clear_accumulator(index);
            self.accumulate(index, terms, 0..self.partitions);

            if self.outputs[index].accumulator_heard {
                self.inverse_into_time(index);
                output.copy_from_slice(&self.time[block..]);
                heard = true;
            } else {
                output.fill(0.);
            }
        }

        heard
    }

    /// Plays the result due in this quantum and does this quantum's share of
    /// the next block's result
    fn process_tail(
        &mut self,
        lanes: [Option<&[f32]>; LANES],
        terms: &'static [Terms],
        outputs: &mut [&mut [f32]],
    ) -> bool {
        let range = self.quantum * RENDER_QUANTUM_SIZE..(self.quantum + 1) * RENDER_QUANTUM_SIZE;

        for (lane, input) in self.lanes.iter_mut().zip(lanes) {
            lane.incoming_heard |= receive(&mut lane.incoming[range.clone()], input);
        }

        let mut heard = false;
        for (output, state) in outputs.iter_mut().zip(&self.outputs) {
            if state.ready_heard {
                output
                    .iter_mut()
                    .zip(&state.ready[range.clone()])
                    .for_each(|(o, r)| *o += r);
                heard = true;
            }
        }

        if self.quantum == 0 {
            self.push_spectra();
            self.terms = terms;
            for index in 0..OUTPUTS {
                self.clear_accumulator(index);
            }
        }

        let partitions = self.quantum * self.partitions / self.quanta
            ..(self.quantum + 1) * self.partitions / self.quanta;
        let latched = self.terms;
        for (index, &terms) in latched.iter().enumerate() {
            self.accumulate(index, terms, partitions.clone());
        }

        if self.quantum == self.quanta - 1 {
            for index in 0..self.terms.len() {
                if self.outputs[index].accumulator_heard {
                    self.inverse_into_time(index);
                    let output = &mut self.outputs[index];
                    output.next.copy_from_slice(&self.time[self.block..]);
                    output.next_heard = true;
                } else {
                    self.outputs[index].next_heard = false;
                }
            }
        }

        self.quantum += 1;
        if self.quantum == self.quanta {
            self.quantum = 0;

            for output in &mut self.outputs {
                std::mem::swap(&mut output.ready, &mut output.next);
                output.ready_heard = output.next_heard;
                output.next_heard = false;
            }

            let block = self.block;
            for lane in &mut self.lanes {
                lane.window.copy_within(block.., 0);
                lane.window[block..].copy_from_slice(&lane.incoming);
                lane.window_heard = [lane.window_heard[1], lane.incoming_heard];
                lane.incoming_heard = false;
            }
        }

        heard
    }

    /// Transform each lane's window into the next slot of its delay line
    fn push_spectra(&mut self) {
        let bins = self.bins();
        for lane in &mut self.lanes {
            lane.newest = (lane.newest + 1) % self.partitions;
            let heard = lane.window_heard[0] || lane.window_heard[1];
            lane.spectra_heard[lane.newest] = heard;

            if heard {
                self.time.copy_from_slice(&lane.window);
                let spectrum = &mut lane.spectra[lane.newest * bins..(lane.newest + 1) * bins];
                self.forward
                    .process_with_scratch(&mut self.time, spectrum, &mut self.scratch)
                    .unwrap();
            }
        }
    }

    fn clear_accumulator(&mut self, output: usize) {
        let output = &mut self.outputs[output];
        if output.accumulator_heard {
            output.accumulator.fill(Complex::default());
            output.accumulator_heard = false;
        }
    }

    /// Add `impulse partition p x input spectrum k - p` over the given
    /// partitions of each term into an output's accumulator
    fn accumulate(&mut self, output: usize, terms: Terms, partitions: std::ops::Range<usize>) {
        let bins = self.bins();
        let state = &mut self.outputs[output];

        for &(lane, channel) in terms {
            let lane = &self.lanes[lane];
            let impulse = &self.impulse[channel];

            for partition in partitions.clone() {
                let slot = (lane.newest + self.partitions - partition) % self.partitions;
                if !lane.spectra_heard[slot] {
                    continue;
                }

                let h = &impulse[partition * bins..(partition + 1) * bins];
                let x = &lane.spectra[slot * bins..(slot + 1) * bins];
                state
                    .accumulator
                    .iter_mut()
                    .zip(h.iter().zip(x))
                    .for_each(|(a, (h, x))| {
                        a.re += h.re * x.re - h.im * x.im;
                        a.im += h.re * x.im + h.im * x.re;
                    });
                state.accumulator_heard = true;
            }
        }
    }

    /// Inverse transform an output's accumulator into `time`; its second
    /// half is the result for the block
    fn inverse_into_time(&mut self, output: usize) {
        let accumulator = &mut self.outputs[output].accumulator;
        // a real signal has no imaginary part at DC and Nyquist
        accumulator[0].im = 0.;
        accumulator[self.block].im = 0.;
        self.inverse
            .process_with_scratch(accumulator, &mut self.time, &mut self.scratch)
            .unwrap();
        // the inverse transform used the accumulator as scratch space
        accumulator.fill(Complex::default());
        self.outputs[output].accumulator_heard = false;
    }

    fn is_silent(&self) -> bool {
        self.lanes.iter().all(|lane| {
            !lane.window_heard[0]
                && !lane.window_heard[1]
                && !lane.incoming_heard
                && lane.spectra_heard.iter().all(|heard| !heard)
        }) && self
            .outputs
            .iter()
            .all(|output| !output.ready_heard && !output.next_heard && !output.accumulator_heard)
    }
}

/// Copy a quantum of input, or zeros for a silent one; returns whether it was
/// heard
fn receive(target: &mut [f32], input: Option<&[f32]>) -> bool {
    match input {
        Some(input) => {
            target.copy_from_slice(input);
            true
        }
        None => {
            target.fill(0.);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    const MONO: &[Terms] = &[&[(0, 0)]];

    fn direct(input: &[f32], impulse: &[f32]) -> Vec<f32> {
        let mut output = vec![0.; input.len() + impulse.len()];
        for (i, x) in input.iter().enumerate().filter(|(_, x)| **x != 0.) {
            for (j, h) in impulse.iter().enumerate() {
                output[i + j] += x * h;
            }
        }
        output
    }

    fn noise(length: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..length)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2. - 1.
            })
            .collect()
    }

    fn render(convolver: &mut Convolver, input: &[f32], quanta: usize) -> Vec<f32> {
        let mut rendered = vec![0.; quanta * RENDER_QUANTUM_SIZE];
        for (index, output) in rendered.chunks_mut(RENDER_QUANTUM_SIZE).enumerate() {
            let start = index * RENDER_QUANTUM_SIZE;
            let lane = (start < input.len()).then(|| &input[start..start + RENDER_QUANTUM_SIZE]);
            convolver.process([lane, None], MONO, &mut [output]);
        }
        rendered
    }

    #[test]
    fn test_matches_direct_convolution_over_every_stage() {
        // reaches into the 8192 stage
        let impulse = noise(20_000, 1);
        // sparse input keeps the direct convolution cheap
        let mut input = vec![0.; 40 * RENDER_QUANTUM_SIZE];
        for i in (0..input.len()).step_by(97) {
            input[i] = noise(1, i as u32 + 7)[0];
        }

        let mut convolver = Convolver::new(std::slice::from_ref(&impulse));
        assert_eq!(convolver.stages.len(), 3);

        let quanta = (input.len() + impulse.len()).div_ceil(RENDER_QUANTUM_SIZE) + 1;
        let rendered = render(&mut convolver, &input, quanta);
        let expected = direct(&input, &impulse);

        assert_float_eq!(rendered[..expected.len()], expected[..], abs_all <= 1e-4);
        assert!(rendered[expected.len()..].iter().all(|v| v.abs() <= 1e-4));
    }

    #[test]
    fn test_short_impulse_is_head_only() {
        let impulse = noise(300, 3);
        let mut input = vec![0.; 4 * RENDER_QUANTUM_SIZE];
        input[5] = 1.;
        input[200] = -0.5;

        let mut convolver = Convolver::new(std::slice::from_ref(&impulse));
        assert_eq!(convolver.stages.len(), 1);

        let rendered = render(&mut convolver, &input, 8);
        let expected = direct(&input, &impulse);
        assert_float_eq!(rendered[..expected.len()], expected[..], abs_all <= 1e-6);
    }

    #[test]
    fn test_silent_after_ringing_out() {
        let impulse = noise(5_000, 5);
        let mut convolver = Convolver::new(&[impulse]);
        assert!(convolver.is_silent());

        let input = noise(RENDER_QUANTUM_SIZE, 9);
        let mut output = [0.; RENDER_QUANTUM_SIZE];
        assert!(convolver.process([Some(&input), None], MONO, &mut [&mut output]));
        assert!(!convolver.is_silent());

        let mut quanta = 0;
        while !convolver.is_silent() {
            convolver.process([None, None], MONO, &mut [&mut output]);
            quanta += 1;
            assert!(quanta < 200, "never rang out");
        }
        // 5000 frames of impulse plus the 1024 stage's two-block delay
        assert!(quanta * RENDER_QUANTUM_SIZE >= 5_000);

        assert!(!convolver.process([None, None], MONO, &mut [&mut output]));
        assert!(output.iter().all(|v| *v == 0.));
    }

    #[test]
    fn test_true_stereo_terms() {
        let impulse: Vec<_> = (0..4).map(|seed| noise(3_000, seed + 11)).collect();
        let left = noise(2 * RENDER_QUANTUM_SIZE, 21);
        let right = noise(2 * RENDER_QUANTUM_SIZE, 22);

        const TRUE_STEREO: &[Terms] = &[&[(0, 0), (1, 2)], &[(0, 1), (1, 3)]];
        let mut convolver = Convolver::new(&impulse);

        let quanta = 30;
        let mut out_left = vec![0.; quanta * RENDER_QUANTUM_SIZE];
        let mut out_right = vec![0.; quanta * RENDER_QUANTUM_SIZE];
        for q in 0..quanta {
            let range = q * RENDER_QUANTUM_SIZE..(q + 1) * RENDER_QUANTUM_SIZE;
            let lanes = if q < 2 {
                [Some(&left[range.clone()]), Some(&right[range.clone()])]
            } else {
                [None, None]
            };
            let (l, r) = (&mut out_left[range.clone()], &mut out_right[range]);
            convolver.process(lanes, TRUE_STEREO, &mut [l, r]);
        }

        let sum = |a: Vec<f32>, b: Vec<f32>| -> Vec<f32> {
            a.iter().zip(&b).map(|(a, b)| a + b).collect()
        };
        let expected_left = sum(direct(&left, &impulse[0]), direct(&right, &impulse[2]));
        let expected_right = sum(direct(&left, &impulse[1]), direct(&right, &impulse[3]));
        let n = expected_left.len();
        assert_float_eq!(out_left[..n], expected_left[..], abs_all <= 1e-4);
        assert_float_eq!(out_right[..n], expected_right[..], abs_all <= 1e-4);
    }
}
