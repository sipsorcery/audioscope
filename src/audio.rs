use std::thread;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use portaudio::{
    self,
    PortAudio,
    Stream,
    NonBlocking,
    Input,
    StreamParameters,
    InputStreamSettings,
    InputStreamCallbackArgs,
    Continue,
    StreamCallbackResult
};
use portaudio::stream::{ InputCallbackArgs};
use num::complex::Complex;
use rustfft::FFT;

use config::Config;
use display::Vec4;

use std::fs::File;
use std::io::Write; 
use std::time::{Duration, SystemTime};

pub type MultiBuffer = Arc<Vec<Mutex<AudioBuffer>>>;
pub type PortAudioStream = Stream<NonBlocking, Input<f32>>;
pub type FilterFunction = Box<dyn FnMut(f32) -> f32 + Send + 'static>;
pub type AudioContextShared = Arc<Mutex<AudioContext>>;

pub struct AudioBuffer {
    pub rendered: bool,
    pub analytic: Vec<Vec4>,
}

const SAMPLE_RATE: f64 = 44_100.0;
const CHANNELS: i32 = 1;
const INTERLEAVED: bool = true;
const GAIN: f32 = 1.0;
const FFT_SIZE: usize = 1024;
const BUFFER_SIZE: usize = 256;
const NUM_BUFFERS: usize = 3;

pub struct AudioContext {
    analytic_filter: Vec<Complex<f32>>,
    pub time_ring_index: usize,
    pub time_ring_buffer: Vec<Complex<f32>>,
    pub display_buffer_index: usize,
    prev_samples: Vec<Vec4>,
    prev_input: Complex<f32>,
    prev_diff: Complex<f32>
}

pub fn init_audio(config: &Config) -> (AudioContextShared, MultiBuffer) {

   let mut output_buffers = Vec::with_capacity(NUM_BUFFERS);

    for _ in 0..NUM_BUFFERS {
        output_buffers.push(Mutex::new(AudioBuffer {
            rendered: true,
            analytic: vec![Vec4 {vec: [0.0, 0.0, 0.0, 0.0]};BUFFER_SIZE + 3],
        }));
    }

    let mut n = FFT_SIZE;
    if n % 2 == 0 {
        n -= 1;
    }
    
    let analytic = make_analytic(n,FFT_SIZE);

    let context = AudioContext {
        analytic_filter: analytic,
        time_ring_index: 0,
        time_ring_buffer: vec![Complex::new(0.0, 0.0); 2 * FFT_SIZE],
        display_buffer_index: 0,
        prev_samples: vec![Vec4 {vec: [0.0, 0.0, 0.0, 0.0]}; 3],
        prev_input: Complex::new(0.0, 0.0),
        prev_diff: Complex::new(0.0, 0.0)
    };

    let context_shared = Arc::new(Mutex::new(context));
    let display_buffer = Arc::new(output_buffers);

    (context_shared, display_buffer)
}

pub fn init_portaudio(
    config: &Config, 
    context_shared: &mut AudioContextShared,
    display_buffers: & MultiBuffer) -> Result<PortAudioStream, portaudio::Error> {

    let pa = try!(PortAudio::new());

    let def_input = try!(pa.default_input_device());
    let input_info = try!(pa.device_info(def_input));
    println!("Default input device name: {}", input_info.name);

    let latency = input_info.default_low_input_latency;
    let input_params = StreamParameters::<f32>::new(def_input, CHANNELS, INTERLEAVED, latency);

    try!(pa.is_input_format_supported(input_params, SAMPLE_RATE));
    let settings = InputStreamSettings::new(input_params, SAMPLE_RATE, BUFFER_SIZE as u32);
    
    let mut angle_lp = get_lowpass(config.audio.cutoff, config.audio.q);
    let mut noise_lp = get_lowpass(0.05, 0.7);

    let mut callback_audio_ctx = context_shared.clone();
    let callback_display_buffer = display_buffers.clone();

    let callback = move |InputStreamCallbackArgs { buffer: data, .. }| {
    
        let mut audio_sample  = vec![Complex::new(0.0f32, 0.0); BUFFER_SIZE];
        for i in 0..BUFFER_SIZE {
            audio_sample[i].re = data[i];
        }

        process_sample(&mut callback_audio_ctx, &audio_sample, &callback_display_buffer, &mut angle_lp, &mut noise_lp);

        Continue
    };

    let stream = try!(pa.open_non_blocking_stream(settings, callback));

    Ok(stream)
}

pub fn get_sample(freq: f32) -> Vec<Complex<f32>> {
    let mut audio_sample  = vec![Complex::new(0.0f32, 0.0); BUFFER_SIZE];
    // Generating a dummy sample.
    for i in 0..BUFFER_SIZE {
        let re_sum: f32 =  (2.0 * std::f32::consts::PI * (i as f32 / freq)).sin() as f32;
        audio_sample[i].re = re_sum;
    }
    audio_sample
}

pub fn process_sample(
    context_shared: &mut AudioContextShared, 
    audio_sample: &Vec<Complex<f32>>,
    display_buffers: &MultiBuffer, 
    angle_low_pass: &mut FilterFunction,
    noise_low_pass: &mut FilterFunction) {

    //println!("process sample {} bytes", audio_sample.len());

    let mut context = context_shared.lock().unwrap();

    let gain = GAIN; //config.audio.gain;
    let mut analytic_buffer = vec![Vec4 {vec: [0.0, 0.0, 0.0, 0.0]}; BUFFER_SIZE + 3];

    // this gets multiplied to convolve stuff
    let mut complex_freq_buffer = vec![Complex::new(0.0f32, 0.0); FFT_SIZE];
    let mut complex_analytic_buffer = vec![Complex::new(0.0f32, 0.0); FFT_SIZE];

    let mut time_ring_index = context.time_ring_index;
    //let display_buffer_index = context.display_buffer_index;
    let time_ring_buffers = &mut context.time_ring_buffer;

    let mut fft = FFT::new(FFT_SIZE, false);
    let mut ifft = FFT::new(FFT_SIZE, true);
    //let mut prev_input = Complex::new(0.0, 0.0); // sample n-1
    //let mut prev_diff = Complex::new(0.0, 0.0); // sample n-1 - sample n-2

    //println!("time ring index {}, display buffer index {}.", time_ring_index, display_buffer_index);

    // Copying the input audio sample into a ring buffer.
    let (left, right) = time_ring_buffers.split_at_mut(FFT_SIZE);
    for ((x, t0), t1) in audio_sample.iter()
        .zip(left[time_ring_index..(time_ring_index + BUFFER_SIZE)].iter_mut())
        .zip(right[time_ring_index..(time_ring_index + BUFFER_SIZE)].iter_mut())
    {
        let mono = Complex::new(gain * x.re, 0.0);
        *t0 = mono;
        *t1 = mono;
    }
    time_ring_index = (time_ring_index + BUFFER_SIZE as usize) % FFT_SIZE;

    // Start the audio sample processing using the oldest sample in the ring buffer:
    // - FFT to extract the frequencies,
    // - Apply the custom analytic filter (phase shift, filtering, windowing etc.),
    // - Inverse FFT to get back the time processed series.

    fft.process(&time_ring_buffers[time_ring_index..time_ring_index + FFT_SIZE], &mut complex_freq_buffer[..]);

    let analytic_filter = &context.analytic_filter;
    for (x, y) in analytic_filter.iter().zip(complex_freq_buffer.iter_mut()) {
        *y = *x * *y;
    }

    ifft.process(&complex_freq_buffer[..], &mut complex_analytic_buffer[..]);

    // Final step is to calculate the angles between each time series data point.
    // This angle is used to set the colour of the line drawn on the OpenGL surface.

    //analytic_buffer[0] = context.prev_samples[0];
    //analytic_buffer[1] = context.prev_samples[1];
    //analytic_buffer[2] = context.prev_samples[2];

    let scale = FFT_SIZE as f32;
    let mut prev_input = context.prev_input;
    let mut prev_diff = context.prev_diff;

    for (&x, y) in complex_analytic_buffer[FFT_SIZE - BUFFER_SIZE..].iter()
        .zip(analytic_buffer[3..].iter_mut()) {

        let diff = x - prev_input; // vector
        prev_input = x;

        let angle = get_angle(diff, prev_diff).abs().log2().max(-1.0e12); // angular velocity (with processing)
        prev_diff = diff;

        let output = angle_low_pass(angle);

        //println!("angle {}, output {}", angle, output);

        *y = Vec4 { vec: [
            x.re / scale,
            x.im / scale,
            output.exp2(), // smoothed angular velocity
            noise_low_pass((angle - output).abs()), // average angular noise
        ]};
    }

    // Write the results to the display buffer.
    let mut display_buffer = display_buffers[context.display_buffer_index].lock().unwrap();
    display_buffer.analytic.copy_from_slice(&analytic_buffer);
    display_buffer.rendered = false;
 
    // Record the last 3 samples to use in the next result.
    //&context.prev_samples.copy_from_slice(&analytic_buffer[BUFFER_SIZE..]);
    context.time_ring_index = time_ring_index;
    context.display_buffer_index = (context.display_buffer_index + 1) % NUM_BUFFERS;
    context.prev_input = prev_input;
    context.prev_diff = prev_diff;
}

// angle between two complex numbers
// scales into [0, 0.5]
fn get_angle(v: Complex<f32>, u: Complex<f32>) -> f32 {
    // 2 atan2(len(len(v)*u − len(u)*v), len(len(v)*u + len(u)*v))
    let len_v_mul_u = v.norm_sqr().sqrt() * u;
    let len_u_mul_v = u.norm_sqr().sqrt() * v;
    let left = (len_v_mul_u - len_u_mul_v).norm_sqr().sqrt(); // this is positive
    let right = (len_v_mul_u + len_u_mul_v).norm_sqr().sqrt(); // this is positive
    left.atan2(right) / ::std::f32::consts::PI
}

// returns biquad lowpass filter
pub fn get_lowpass(n: f32, q: f32) -> FilterFunction {
    let k = (0.5 * n * ::std::f32::consts::PI).tan();
    let norm = 1.0 / (1.0 + k / q + k * k);
    let a0 = k * k * norm;
    let a1 = 2.0 * a0;
    let a2 = a0;
    let b1 = 2.0 * (k * k - 1.0) * norm;
    let b2 = (1.0 - k / q + k * k) * norm;

    let mut w1 = 0.0;
    let mut w2 = 0.0;
    // \ y[n]=b_{0}w[n]+b_{1}w[n-1]+b_{2}w[n-2],
    // where
    // w[n]=x[n]-a_{1}w[n-1]-a_{2}w[n-2].
    Box::new(move |x| {
        let w0 = x - b1 * w1 - b2 * w2;
        let y = a0 * w0 + a1 * w1 + a2 * w2;
        w2 = w1;
        w1 = w0;
        y
    })
}

// FIR analytical signal transform of length n with zero padding to be length m
// real part removes DC and nyquist, imaginary part phase shifts by 90
// should act as bandpass (remove all negative frequencies + DC & nyquist)
fn make_analytic(n: usize, m: usize) -> Vec<Complex<f32>> {
    use ::std::f32::consts::PI;
    assert_eq!(n % 2, 1, "n should be odd");
    assert!(n <= m, "n should be less than or equal to m");
    // let a = 2.0 / n as f32;
    let mut fft = FFT::new(m, false);

    let mut impulse = vec![Complex::new(0.0, 0.0); m];
    let mut freqs = impulse.clone();

    let mid = (n - 1) / 2;

    impulse[mid].re = 1.0;
    let re = -1.0 / (mid - 1) as f32;
    for i in 1..mid+1 {
        if i % 2 == 0 {
            impulse[mid + i].re = re;
            impulse[mid - i].re = re;
        } else {
            let im = 2.0 / PI / i as f32;
            impulse[mid + i].im = im;
            impulse[mid - i].im = -im;
        }
        // hamming window
        let k = 0.53836 + 0.46164 * (i as f32 * PI / (mid + 1) as f32).cos();
        impulse[mid + i] = impulse[mid + i].scale(k);
        impulse[mid - i] = impulse[mid - i].scale(k);
    }
    fft.process(&impulse, &mut freqs);
    freqs
}
#[test]
fn test_analytic() {
    let m = 1024; // ~ 40hz
    let n = m / 4 * 3 - 1; // overlap 75%
    let freqs = make_analytic(n, m);
    // DC is below -6db
    assert!(10.0 * freqs[0].norm_sqr().log(10.0) < -6.0);
    // 40hz is above 0db
    assert!(10.0 * freqs[1].norm_sqr().log(10.0) > 0.0);
    // -40hz is below -12db
    assert!(10.0 * freqs[m-1].norm_sqr().log(10.0) < -12.0);
    // actually these magnitudes are halved bc passband is +6db
}

#[test]
fn test_lowpass() {
    let mut lp = get_lowpass(0.5, 0.71);
    println!();
    println!("{}", lp(1.0));
    for _ in 0..10 {
        println!("{}", lp(0.0));
    }
    for _ in 0..10 {
        assert!(lp(0.0).abs() < 0.5); // if it's unstable, it'll be huge
    }
}
