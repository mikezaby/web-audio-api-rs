use std::any::Any;

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::convolution::{Convolver, Terms};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};

use super::{AudioNode, AudioNodeOptions, ChannelConfig, ChannelCountMode, ChannelInterpretation};

/// Scale buffer by an equal-power normalization
// see - <https://webaudio.github.io/web-audio-api/#dom-convolvernode-normalize>
fn normalize_buffer(buffer: &AudioBuffer) -> f32 {
    let gain_calibration = 0.00125;
    let gain_calibration_sample_rate = 44100.;
    let min_power = 0.000125;

    // Normalize by RMS power.
    let number_of_channels = buffer.number_of_channels();
    let length = buffer.length();
    let sample_rate = buffer.sample_rate();

    let mut power: f32 = buffer
        .channels()
        .iter()
        .map(|c| c.as_slice().iter().map(|&s| s * s).sum::<f32>())
        .sum();

    power = (power / (number_of_channels * length) as f32).sqrt();

    // Protect against accidental overload.
    if !power.is_finite() || power.is_nan() || power < min_power {
        power = min_power;
    }

    let mut scale = 1. / power;

    // Calibrate to make perceived volume same as unprocessed.
    scale *= gain_calibration;

    // Scale depends on sample-rate.
    scale *= gain_calibration_sample_rate / sample_rate;

    // True-stereo compensation.
    if number_of_channels == 4 {
        scale *= 0.5;
    }

    scale
}

/// `ConvolverNode` options
//dictionary ConvolverOptions : AudioNodeOptions {
//  AudioBuffer? buffer;
//  boolean disableNormalization = false;
//};
#[derive(Clone, Debug)]
pub struct ConvolverOptions {
    /// The desired buffer for the ConvolverNode
    pub buffer: Option<AudioBuffer>,
    /// The opposite of the desired initial value for the normalize attribute
    pub disable_normalization: bool,
    /// AudioNode options
    pub audio_node_options: AudioNodeOptions,
}

impl Default for ConvolverOptions {
    fn default() -> Self {
        Self {
            buffer: None,
            disable_normalization: false,
            audio_node_options: AudioNodeOptions {
                channel_count: 2,
                channel_count_mode: ChannelCountMode::ClampedMax,
                channel_interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
}

/// Assert that the channel count is valid for the ConvolverNode
/// see <https://webaudio.github.io/web-audio-api/#audionode-channelcount-constraints>
///
/// # Panics
///
/// This function panics if given count is greater than 2
///
#[track_caller]
#[inline(always)]
fn assert_valid_channel_count(count: usize) {
    assert!(
        count <= 2,
        "NotSupportedError - ConvolverNode channel count cannot be greater than two"
    );
}

/// Assert that the channel count mode is valid for the ConvolverNode
/// see <https://webaudio.github.io/web-audio-api/#audionode-channelcountmode-constraints>
///
/// # Panics
///
/// This function panics if given count mode is [`ChannelCountMode::Max`]
///
#[track_caller]
#[inline(always)]
fn assert_valid_channel_count_mode(mode: ChannelCountMode) {
    assert_ne!(
        mode,
        ChannelCountMode::Max,
        "NotSupportedError - ConvolverNode channel count mode cannot be set to max"
    );
}

/// Processing node which applies a linear convolution effect given an impulse response.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/ConvolverNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#ConvolverNode>
/// - see also: [`BaseAudioContext::create_convolver`]
///
/// The current implementation only handles mono-to-mono convolutions. The provided impulse
/// response buffer and the input signal will be downmixed appropriately.
///
/// # Usage
///
/// ```no_run
/// use std::fs::File;
///
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ConvolverNode, ConvolverOptions};
///
/// let context = AudioContext::default();
/// let file = File::open("samples/vocals-dry.wav").unwrap();
/// let audio_buffer = context.decode_audio_data_sync(file).unwrap();
///
/// let impulse_file = File::open("samples/small-room-response.wav").unwrap();
/// let impulse_buffer = context.decode_audio_data_sync(impulse_file).unwrap();
///
/// let mut src = context.create_buffer_source();
/// src.set_buffer(audio_buffer);
///
/// let mut convolve = ConvolverNode::new(&context, ConvolverOptions::default());
/// convolve.set_buffer(impulse_buffer);
///
/// src.connect(&convolve);
/// convolve.connect(&context.destination());
/// src.start();
/// std::thread::sleep(std::time::Duration::from_millis(4_000));
/// ```
///
/// # Examples
///
/// - `cargo run --release --example convolution`
///
#[derive(Debug)]
pub struct ConvolverNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Info about audio node channel configuration
    channel_config: ChannelConfig,
    /// Perform equal power normalization on response buffer
    normalize: bool,
    /// The response buffer, nullable
    buffer: Option<AudioBuffer>,
}

impl AudioNode for ConvolverNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        1
    }

    // see <https://webaudio.github.io/web-audio-api/#audionode-channelcount-constraints>
    fn set_channel_count(&self, count: usize) {
        assert_valid_channel_count(count);
        self.channel_config.set_count(count, self.registration());
    }

    // see <https://webaudio.github.io/web-audio-api/#audionode-channelcountmode-constraints>
    fn set_channel_count_mode(&self, mode: ChannelCountMode) {
        assert_valid_channel_count_mode(mode);
        self.channel_config
            .set_count_mode(mode, self.registration());
    }
}

impl ConvolverNode {
    /// returns a `ConvolverNode` instance
    ///
    /// # Arguments
    ///
    /// * `context` - audio context in which the audio node will live.
    /// * `options` - convolver options
    ///
    /// # Panics
    ///
    /// Panics when an AudioBuffer is provided via the `ConvolverOptions` with a sample rate
    /// different from the audio context sample rate.
    pub fn new<C: BaseAudioContext>(context: &C, options: ConvolverOptions) -> Self {
        let ConvolverOptions {
            buffer,
            disable_normalization,
            audio_node_options,
        } = options;

        assert_valid_channel_count(audio_node_options.channel_count);
        assert_valid_channel_count_mode(audio_node_options.channel_count_mode);

        let mut node = context.base().register(move |registration| {
            let renderer = ConvolverRenderer {
                convolver: None,
                impulse_number_of_channels: 0,
                input_number_of_channels: 1,
            };

            let node = Self {
                registration,
                channel_config: audio_node_options.into(),
                normalize: !disable_normalization,
                buffer: None,
            };

            (node, Box::new(renderer))
        });

        // renderer has been sent to render thread, we can send it messages
        if let Some(buffer) = buffer {
            node.set_buffer(buffer);
        }

        node
    }

    /// Get the current impulse response buffer
    pub fn buffer(&self) -> Option<&AudioBuffer> {
        self.buffer.as_ref()
    }

    /// Set or update the impulse response buffer
    ///
    /// # Panics
    ///
    /// Panics when the sample rate of the provided AudioBuffer differs from the audio context
    /// sample rate.
    pub fn set_buffer(&mut self, buffer: AudioBuffer) {
        // If the buffer number of channels is not 1, 2, 4, or if the sample-rate of the buffer is
        // not the same as the sample-rate of its associated BaseAudioContext, a NotSupportedError
        // MUST be thrown.

        let sample_rate = buffer.sample_rate();
        assert_eq!(
            sample_rate,
            self.context().sample_rate(),
            "NotSupportedError - sample rate of the convolution buffer must match the audio context"
        );

        let number_of_channels = buffer.number_of_channels();
        assert!(
            [1, 2, 4].contains(&number_of_channels),
            "NotSupportedError - the convolution buffer must consist of 1, 2 or 4 channels"
        );

        let msg = ConvolverInfosMessage::new(&buffer, self.normalize);
        self.registration.post_message(msg);
        self.buffer = Some(buffer);
    }

    /// Denotes if the response buffer will be scaled with an equal-power normalization
    pub fn normalize(&self) -> bool {
        self.normalize
    }

    /// Update the `normalize` setting. This will only have an effect when `set_buffer` is called.
    pub fn set_normalize(&mut self, value: bool) {
        self.normalize = value;
    }
}

struct ConvolverInfosMessage {
    convolver: Option<Convolver>,
    impulse_number_of_channels: usize,
}

impl ConvolverInfosMessage {
    /// Prepare the convolution of an impulse response, on the control thread
    fn new(buffer: &AudioBuffer, normalize: bool) -> Self {
        // normalize before padding because the length of the buffer affects the scale
        let scale = if normalize {
            normalize_buffer(buffer)
        } else {
            1.
        };

        let impulse: Vec<Vec<f32>> = (0..buffer.number_of_channels())
            .map(|channel| {
                buffer
                    .get_channel_data(channel)
                    .iter()
                    .map(|v| v * scale)
                    .collect()
            })
            .collect();

        Self {
            convolver: Some(Convolver::new(&impulse)),
            impulse_number_of_channels: buffer.number_of_channels(),
        }
    }
}

/// The (lane, impulse channel) terms of each output channel per input and
/// impulse channel count. Lane 0 takes input channel 0, lane 1 input channel
/// 1, or channel 0 again for a mono input that has two outputs.
// cf. https://webaudio.github.io/web-audio-api/#Convolution-channel-configurations
fn routing(input_channels: usize, impulse_channels: usize) -> &'static [Terms] {
    match (input_channels, impulse_channels) {
        (1, 1) => &[&[(0, 0)]],
        (1, 2) | (2, 2) => &[&[(0, 0)], &[(1, 1)]],
        (2, 1) => &[&[(0, 0)], &[(1, 0)]],
        // true stereo; a mono input is upmixed to stereo first
        (1, 4) | (2, 4) => &[&[(0, 0), (1, 2)], &[(0, 1), (1, 3)]],
        _ => unreachable!(),
    }
}

struct ConvolverRenderer {
    convolver: Option<Convolver>,
    impulse_number_of_channels: usize,
    /// channel count of the last input that was heard, so a silent quantum
    /// (a single channel) does not cut the other channels' tail
    input_number_of_channels: usize,
}

impl AudioProcessor for ConvolverRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        let convolver = match &mut self.convolver {
            None => {
                // no convolution buffer set, passthrough
                output.clone_from(input);
                return !input.is_silent();
            }
            Some(convolver) => convolver,
        };

        // return early if the input is silent and the convolution rang out
        if input.is_silent() {
            if convolver.is_silent() {
                output.make_silent();
                return false;
            }
        } else {
            self.input_number_of_channels = input.number_of_channels();
        }

        let terms = routing(
            self.input_number_of_channels,
            self.impulse_number_of_channels,
        );

        let lanes = if input.is_silent() {
            [None, None]
        } else if input.number_of_channels() == 1 {
            let mono = &input.channel_data(0)[..];
            [Some(mono), (terms.len() > 1).then_some(mono)]
        } else {
            [
                Some(&input.channel_data(0)[..]),
                Some(&input.channel_data(1)[..]),
            ]
        };

        output.set_number_of_channels(terms.len());
        let heard = match output.channels_mut() {
            [mono] => convolver.process(lanes, terms, &mut [&mut mono[..]]),
            [left, right] => convolver.process(lanes, terms, &mut [&mut left[..], &mut right[..]]),
            _ => unreachable!(),
        };

        if !heard {
            output.make_silent();
        }

        true
    }

    fn onmessage(&mut self, msg: &mut dyn Any) {
        if let Some(msg) = msg.downcast_mut::<ConvolverInfosMessage>() {
            let ConvolverInfosMessage {
                convolver,
                impulse_number_of_channels,
            } = msg;
            // Avoid deallocation in the render thread by swapping the convolver.
            std::mem::swap(&mut self.convolver, convolver);
            self.impulse_number_of_channels = *impulse_number_of_channels;

            return;
        }

        log::warn!("ConvolverRenderer: Dropping incoming message {msg:?}");
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::{AudioNodeId, BaseAudioContext, OfflineAudioContext};
    use crate::node::{AudioBufferSourceNode, AudioBufferSourceOptions, AudioScheduledSourceNode};
    use crate::render::{Alloc, NodeCollection};

    use super::*;

    #[test]
    #[should_panic]
    fn test_buffer_sample_rate_matches() {
        let context = OfflineAudioContext::new(1, 128, 44100.);

        let ir = vec![1.];
        let ir = AudioBuffer::from(vec![ir; 1], 48000.); // sample_rate differs
        let options = ConvolverOptions {
            buffer: Some(ir),
            ..ConvolverOptions::default()
        };

        let _ = ConvolverNode::new(&context, options);
    }

    #[test]
    #[should_panic]
    fn test_buffer_must_have_1_2_4_channels() {
        let context = OfflineAudioContext::new(1, 128, 48000.);

        let ir = vec![1.];
        let ir = AudioBuffer::from(vec![ir; 3], 48000.); // three channels
        let options = ConvolverOptions {
            buffer: Some(ir),
            ..ConvolverOptions::default()
        };

        let _ = ConvolverNode::new(&context, options);
    }

    #[test]
    fn test_constructor_options_buffer() {
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(1, 10, sample_rate);

        let ir = vec![1.];
        let calibration = 0.00125;
        let channel_data = vec![0., 1., 0., -1., 0.];
        let expected = [0., calibration, 0., -calibration, 0., 0., 0., 0., 0., 0.];

        // identity ir
        let ir = AudioBuffer::from(vec![ir; 1], sample_rate);
        let options = ConvolverOptions {
            buffer: Some(ir),
            ..ConvolverOptions::default()
        };
        let conv = ConvolverNode::new(&context, options);
        conv.connect(&context.destination());

        let buffer = AudioBuffer::from(vec![channel_data; 1], sample_rate);
        let mut src = context.create_buffer_source();
        src.connect(&conv);
        src.set_buffer(buffer);
        src.start();

        let output = context.start_rendering_sync();

        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    fn test_convolve(signal: &[f32], impulse_resp: Option<Vec<f32>>, length: usize) -> AudioBuffer {
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(1, length, sample_rate);

        let input = AudioBuffer::from(vec![signal.to_vec()], sample_rate);
        let mut src = AudioBufferSourceNode::new(&context, AudioBufferSourceOptions::default());
        src.set_buffer(input);
        src.start();

        let mut conv = ConvolverNode::new(&context, ConvolverOptions::default());
        if let Some(ir) = impulse_resp {
            conv.set_buffer(AudioBuffer::from(vec![ir.to_vec()], sample_rate));
        }

        src.connect(&conv);
        conv.connect(&context.destination());

        context.start_rendering_sync()
    }

    #[test]
    fn test_passthrough() {
        let output = test_convolve(&[0., 1., 0., -1., 0.], None, 10);
        let expected = [0., 1., 0., -1., 0., 0., 0., 0., 0., 0.];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_empty() {
        let ir = vec![];
        let output = test_convolve(&[0., 1., 0., -1., 0.], Some(ir), 10);
        let expected = [0.; 10];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_zeroed() {
        let ir = vec![0., 0., 0., 0., 0., 0.];
        let output = test_convolve(&[0., 1., 0., -1., 0.], Some(ir), 10);
        let expected = [0.; 10];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_identity() {
        let ir = vec![1.];
        let calibration = 0.00125;
        let output = test_convolve(&[0., 1., 0., -1., 0.], Some(ir), 10);
        let expected = [0., calibration, 0., -calibration, 0., 0., 0., 0., 0., 0.];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_two_id() {
        let ir = vec![1., 1.];
        let calibration = 0.00125;
        let output = test_convolve(&[0., 1., 0., -1., 0.], Some(ir), 10);
        let expected = [
            0.,
            calibration,
            calibration,
            -calibration,
            -calibration,
            0.,
            0.,
            0.,
            0.,
            0.,
        ];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_should_have_tail_time() {
        // impulse response of length 256
        const IR_LEN: usize = 256;
        let ir = vec![1.; IR_LEN];

        // unity input signal
        let input = &[1.];

        // render into a buffer of size 512
        let output = test_convolve(input, Some(ir), 512);

        // we expect non-zero output in the range 0 to IR_LEN
        let output = output.channel_data(0).as_slice();
        assert!(!output[..IR_LEN].iter().any(|v| *v <= 1E-6));
        assert_float_eq!(&output[IR_LEN..], &[0.; 512 - IR_LEN][..], abs_all <= 1E-6);
    }

    #[test]
    fn test_fresh_impulse_has_no_tail() {
        let alloc = Alloc::with_capacity(2);
        let nodes = NodeCollection::new();
        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: std::cell::Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        let ir = AudioBuffer::from(vec![vec![1.; 48000]], 48000.);
        let mut renderer = ConvolverRenderer {
            convolver: None,
            impulse_number_of_channels: 0,
            input_number_of_channels: 1,
        };
        renderer.onmessage(&mut ConvolverInfosMessage::new(&ir, true));

        let inputs = [AudioRenderQuantum::from(alloc.silence())];
        let mut outputs = [AudioRenderQuantum::from(alloc.silence())];

        // a convolver that never heard its input stays silent at once
        let tail = renderer.process(
            &inputs,
            &mut outputs,
            AudioParamValues::from(&nodes),
            &scope,
        );
        assert!(!tail);
        assert!(outputs[0].is_silent());

        // once fed, it rings for the length of the impulse
        let mut inputs = [AudioRenderQuantum::from(alloc.silence())];
        inputs[0].channel_data_mut(0)[0] = 1.;
        assert!(renderer.process(
            &inputs,
            &mut outputs,
            AudioParamValues::from(&nodes),
            &scope
        ));

        let inputs = [AudioRenderQuantum::from(alloc.silence())];
        assert!(renderer.process(
            &inputs,
            &mut outputs,
            AudioParamValues::from(&nodes),
            &scope
        ));
        assert!(!outputs[0].is_silent());
    }

    #[test]
    fn test_channel_config_1_chan_in_1_chan_ir() {
        let number_of_channels = 1;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1.]], sample_rate);
        let ir = AudioBuffer::from(vec![vec![0., 1.]], sample_rate);

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected = [0.; 128];
        expected[1] = 1.;

        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_1_chan_in_2_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1.]], sample_rate);
        let ir = AudioBuffer::from(vec![vec![0., 1., 0.], vec![0., 0., 1.]], sample_rate);

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[2] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_2_chan_in_1_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1., 0.], vec![0., 1.]], sample_rate);
        let ir = AudioBuffer::from(vec![vec![0., 1.]], sample_rate);

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[2] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_2_chan_in_2_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1., 0.], vec![0., 1.]], sample_rate);
        let ir = AudioBuffer::from(vec![vec![0., 1., 0.], vec![0., 0., 1.]], sample_rate);

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[3] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_2_chan_in_4_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1., 0.], vec![0., 1.]], sample_rate);
        let ir = AudioBuffer::from(
            vec![
                vec![0., 1., 0., 0., 0.], // in 0 -> out 0
                vec![0., 0., 1., 0., 0.], // in 0 -> out 1
                vec![0., 0., 0., 1., 0.], // in 1 -> out 0
                vec![0., 0., 0., 0., 1.], // in 1 -> out 1
            ],
            sample_rate,
        );

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;
        expected_left[4] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[2] = 1.;
        expected_right[5] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_1_chan_in_4_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1., 0.]], sample_rate);
        let ir = AudioBuffer::from(
            vec![
                vec![0., 1., 0., 0., 0.], // in 0 -> out 0
                vec![0., 0., 1., 0., 0.], // in 0 -> out 1
                vec![0., 0., 0., 1., 0.], // in 0 -> out 0
                vec![0., 0., 0., 0., 1.], // in 0 -> out 1
            ],
            sample_rate,
        );

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;
        expected_left[3] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[2] = 1.;
        expected_right[4] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }
}
