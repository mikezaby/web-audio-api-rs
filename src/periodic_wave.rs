//! PeriodicWave interface

use std::sync::Arc;

use num_complex::Complex;
use realfft::RealFftPlanner;

use crate::context::BaseAudioContext;

/// Options for constructing a [`PeriodicWave`]
#[derive(Debug, Default, Clone)]
pub struct PeriodicWaveOptions {
    /// The real parameter represents an array of cosine terms of Fourier series.
    ///
    /// The first element (index 0) represents the DC-offset.
    /// This offset has to be given but will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and
    /// harmonics of the periodic waveform.
    pub real: Option<Vec<f32>>,
    /// The imag parameter represents an array of sine terms of Fourier series.
    ///
    /// The first element (index 0) will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and
    /// harmonics of the periodic waveform.
    pub imag: Option<Vec<f32>>,
    /// By default PeriodicWave is build with normalization enabled (disable_normalization = false).
    /// In this case, a peak normalization is applied to the given custom periodic waveform.
    ///
    /// If disable_normalization is enabled (disable_normalization = true), the normalization is
    /// defined by the periodic waveform characteristics (img, and real fields).
    pub disable_normalization: bool,
}

/// `PeriodicWave` represents an arbitrary periodic waveform to be used with an `OscillatorNode`.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/PeriodicWave>
/// - specification: <https://webaudio.github.io/web-audio-api/#PeriodicWave>
/// - see also: [`BaseAudioContext::create_periodic_wave`]
/// - see also: [`OscillatorNode`](crate::node::OscillatorNode)
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::{PeriodicWave, PeriodicWaveOptions};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// let context = AudioContext::default();
///
/// // generate a simple waveform with 2 harmonics
/// let options = PeriodicWaveOptions {
///   real: Some(vec![0., 0., 0.]),
///   imag: Some(vec![0., 0.5, 0.5]),
///   disable_normalization: false,
/// };
///
/// let periodic_wave = PeriodicWave::new(&context, options);
///
/// let mut osc = context.create_oscillator();
/// osc.set_periodic_wave(periodic_wave);
/// osc.connect(&context.destination());
/// osc.start();
/// ```
/// # Examples
///
/// - `cargo run --release --example oscillators`
///
// Basically a wrapper around Arc<Vec<f32>>, so `PeriodicWave`s are cheap to clone
#[derive(Debug, Clone, Default)]
pub struct PeriodicWave {
    wavetable: Arc<Vec<f32>>,
}

const PERIODIC_WAVE_TABLE_LENGTH: usize = 8192;

impl PeriodicWave {
    /// Returns a `PeriodicWave`
    ///
    /// # Arguments
    ///
    /// * `real` - The real parameter represents an array of cosine terms of Fourier series.
    /// * `imag` - The imag parameter represents an array of sine terms of Fourier series.
    /// * `constraints` - The constraints parameter specifies the normalization mode of the `PeriodicWave`
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * `real` is defined and its length is less than 2
    /// * `imag` is defined and its length is less than 2
    /// * `real` and `imag` are defined and theirs lengths are not equal
    /// * `PeriodicWave` is more than 8192 components
    //
    // @notes:
    // - Built-in types of the `OscillatorNode` should use periodic waves
    // c.f. https://webaudio.github.io/web-audio-api/#oscillator-coefficients
    // - The question of bandlimited oscillators should also be handled
    // e.g. https://www.dafx12.york.ac.uk/papers/dafx12_submission_69.pdf
    pub fn new<C: BaseAudioContext>(_context: &C, options: PeriodicWaveOptions) -> Self {
        let PeriodicWaveOptions {
            real,
            imag,
            disable_normalization,
        } = options;

        let (real, imag) = match (real, imag) {
            (Some(r), Some(i)) => {
                assert_eq!(
                    r.len(),
                    i.len(),
                    "IndexSizeError - `real` and `imag` length should be equal"
                );
                assert!(
                    r.len() >= 2,
                    "IndexSizeError - `real` and `imag` length should at least 2"
                );

                (r, i)
            }
            (Some(r), None) => {
                assert!(
                    r.len() >= 2,
                    "IndexSizeError - `real` and `imag` length should at least 2"
                );

                let len = r.len();
                (r, vec![0.; len])
            }
            (None, Some(i)) => {
                assert!(
                    i.len() >= 2,
                    "IndexSizeError - `real` and `imag` length should at least 2"
                );

                let len = i.len();
                (vec![0.; len], i)
            }
            // Defaults to sine wave
            // [spec] Note: When setting this PeriodicWave on an OscillatorNode,
            // this is equivalent to using the built-in type "sine".
            _ => (vec![0., 0.], vec![0., 1.]),
        };

        let normalize = !disable_normalization;
        // [spec] A conforming implementation MUST support PeriodicWave up to at least 8192 elements.
        let wavetable =
            Self::generate_wavetable(&real, &imag, normalize, PERIODIC_WAVE_TABLE_LENGTH);

        Self {
            wavetable: Arc::new(wavetable),
        }
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.wavetable[..]
    }

    // cf. https://webaudio.github.io/web-audio-api/#waveform-generation
    //
    // The table is the sum over k of a[k] cos(2πki/N) + b[k] sin(2πki/N), which is the inverse
    // real FFT of (a[k] - j b[k]) / 2: one transform instead of a sin and a cos per sample and
    // harmonic. A harmonic above N/2 lands on bin N - k with its sine term negated.
    fn generate_wavetable(reals: &[f32], imags: &[f32], normalize: bool, size: usize) -> Vec<f32> {
        let half = size / 2;
        let mut spectrum = vec![Complex::<f64>::default(); half + 1];

        for (k, (&real, &imag)) in reals.iter().zip(imags).enumerate().skip(1) {
            let (real, imag) = (f64::from(real), f64::from(imag));
            match k % size {
                // cos(2πki/N) is 1 and sin(2πki/N) is 0
                0 => spectrum[0].re += real,
                bin if bin < half => {
                    spectrum[bin].re += real / 2.;
                    spectrum[bin].im -= imag / 2.;
                }
                // cos(πi) alternates and sin(πi) is 0
                bin if bin == half => spectrum[half].re += real,
                bin => {
                    spectrum[size - bin].re += real / 2.;
                    spectrum[size - bin].im += imag / 2.;
                }
            }
        }

        let inverse = RealFftPlanner::<f64>::new().plan_fft_inverse(size);
        let mut table = inverse.make_output_vec();
        inverse
            .process(&mut spectrum, &mut table)
            .expect("the spectrum of a real table");
        let mut wavetable: Vec<f32> = table.iter().map(|&v| v as f32).collect();

        if normalize {
            Self::normalize(&mut wavetable);
        }

        wavetable
    }

    fn normalize(wavetable: &mut [f32]) {
        let mut max = 0.;

        for sample in wavetable.iter() {
            let abs = sample.abs();
            if abs > max {
                max = abs;
            }
        }

        // prevent division by 0. (nothing to normalize anyway...)
        if max > 0. {
            let norm_factor = 1. / max;

            for sample in wavetable.iter_mut() {
                *sample *= norm_factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use std::f32::consts::PI;

    use super::{PeriodicWave, PeriodicWaveOptions, PERIODIC_WAVE_TABLE_LENGTH};
    use crate::context::AudioContext;

    #[test]
    #[should_panic]
    fn fails_to_build_when_only_real_is_defined_and_too_short() {
        let context = AudioContext::default();

        let options = PeriodicWaveOptions {
            real: Some(vec![0.]),
            imag: None,
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, options);
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_only_imag_is_defined_and_too_short() {
        let context = AudioContext::default();

        let options = PeriodicWaveOptions {
            real: None,
            imag: Some(vec![0.]),
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, options);
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_and_real_not_equal_length() {
        let context = AudioContext::default();

        let options = PeriodicWaveOptions {
            real: Some(vec![0., 0., 0.]),
            imag: Some(vec![0., 0.]),
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, options);
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_and_real_too_shorts() {
        let context = AudioContext::default();

        let options = PeriodicWaveOptions {
            real: Some(vec![0.]),
            imag: Some(vec![0.]),
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, options);
    }

    #[test]
    fn wavetable_generate_sine() {
        let reals = [0., 0.];
        let imags = [0., 1.];

        let result =
            PeriodicWave::generate_wavetable(&reals, &imags, true, PERIODIC_WAVE_TABLE_LENGTH);
        let mut expected = Vec::new();

        for i in 0..PERIODIC_WAVE_TABLE_LENGTH {
            let sample = (i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();
            expected.push(sample);
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-6);
    }

    #[test]
    fn wavetable_generate_2f_not_norm() {
        let reals = [0., 0., 0.];
        let imags = [0., 0.5, 0.5];

        let result =
            PeriodicWave::generate_wavetable(&reals, &imags, false, PERIODIC_WAVE_TABLE_LENGTH);
        let mut expected = Vec::new();

        for i in 0..PERIODIC_WAVE_TABLE_LENGTH {
            let mut sample = 0.;
            // fundamental frequency
            sample += 0.5 * (1. * i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();
            // 1rst partial
            sample += 0.5 * (2. * i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();

            expected.push(sample);
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-6);
    }

    #[test]
    fn normalize() {
        {
            let mut signal = [-0.5, 0.2];
            PeriodicWave::normalize(&mut signal);
            let expected = [-1., 0.4];

            assert_float_eq!(signal[..], expected[..], abs_all <= 0.);
        }

        {
            let mut signal = [0.5, -0.2];
            PeriodicWave::normalize(&mut signal);
            let expected = [1., -0.4];

            assert_float_eq!(signal[..], expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn wavetable_generate_2f_norm() {
        let reals = [0., 0., 0.];
        let imags = [0., 0.5, 0.5];

        let result =
            PeriodicWave::generate_wavetable(&reals, &imags, true, PERIODIC_WAVE_TABLE_LENGTH);
        let mut expected = Vec::new();

        for i in 0..PERIODIC_WAVE_TABLE_LENGTH {
            let mut sample = 0.;
            // fundamental frequency
            sample += 0.5 * (1. * i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();
            // 1rst partial
            sample += 0.5 * (2. * i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();

            expected.push(sample);
        }

        PeriodicWave::normalize(&mut expected);

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-6);
    }

    #[test]
    fn wavetable_matches_the_sum_with_harmonics_above_half_the_table() {
        // every harmonic up to the 8191st, with a cosine and a sine term each
        let reals: Vec<f32> = (0..8192)
            .map(|k| ((k * 7) % 13) as f32 / 13. - 0.5)
            .collect();
        let imags: Vec<f32> = (0..8192)
            .map(|k| ((k * 5) % 11) as f32 / 11. - 0.5)
            .collect();

        let result =
            PeriodicWave::generate_wavetable(&reals, &imags, false, PERIODIC_WAVE_TABLE_LENGTH);

        let size = PERIODIC_WAVE_TABLE_LENGTH;
        for i in (0..size).step_by(size / 64) {
            let mut expected = 0f64;
            for k in 1..reals.len() {
                let angle = 2. * std::f64::consts::PI * ((i * k) % size) as f64 / size as f64;
                expected += f64::from(reals[k]) * angle.cos() + f64::from(imags[k]) * angle.sin();
            }
            assert_float_eq!(result[i], expected as f32, abs <= 1e-3, "sample {i}");
        }
    }
}
