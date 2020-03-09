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
};
use num::complex::Complex;
use rustfft::FFT;

use config::Config;
use display::Vec4;

use std::fs::File;
use std::io::Write; 

pub type MultiBuffer = Arc<Vec<Mutex<AudioBuffer>>>;
pub type PortAudioStream = Stream<NonBlocking, Input<f32>>;

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

pub struct Context {
    pub analyticFilter: Vec<Complex<f32>>,
    pub displayBuffers: MultiBuffer
}

pub fn init_audio(config: &Config) -> Context {

   let mut timeSeries = Vec::with_capacity(NUM_BUFFERS);

    for _ in 0..NUM_BUFFERS {
        timeSeries.push(Mutex::new(AudioBuffer {
            rendered: true,
            analytic: vec![Vec4 {vec: [0.0, 0.0, 0.0, 0.0]};BUFFER_SIZE + 3],
        }));
    }

    //let mut fBuffer = File::create("dump.txt").expect("Unable to create dumpfile");

    let mut n = FFT_SIZE;
    if n % 2 == 0 {
        n -= 1;
    }
    
    let analytic = make_analytic(n,FFT_SIZE);

    //let mut i =0;
    //for x in analytic.iter() {
    //    println!("{}:{},{}", i, x.re, x.im);
    //    i += 1;
    //}

    //std::process::exit(0);

    let context = Context {
        analyticFilter: analytic,
        displayBuffers: Arc::new(timeSeries)
    };

    context
}

//pub fn generate_sample(analytic_filter: &Vec<Complex<f32>>, shared_buffers: &MultiBuffer, freq: f32) -> f32 {
pub fn generate_sample(shared_buffers: &MultiBuffer, freq: f64) {

    println!("generate_sample freq {}", freq);

    let mut n = FFT_SIZE;
    if n % 2 == 0 {
        n -= 1;
    }
    
    let analytic_filter = make_analytic(n,FFT_SIZE);

    let mut buffer_index = 0;
    let gain = GAIN; //config.audio.gain;
    let buffers = shared_buffers.clone();
    let mut analytic_buffer = vec![Vec4 {vec: [0.0, 0.0, 0.0, 0.0]}; BUFFER_SIZE + 3];

    // this gets multiplied to convolve stuff
    let mut complex_freq_buffer = vec![Complex::new(0.0f32, 0.0); FFT_SIZE];
    let mut complex_analytic_buffer = vec![Complex::new(0.0f32, 0.0); FFT_SIZE];
    let mut data_complex_buffer  = vec![Complex::new(0.0f32, 0.0); FFT_SIZE];

    //let analytic = make_analytic(n, FFT_SIZE);
    let mut fft = FFT::new(FFT_SIZE, false);
    let mut ifft = FFT::new(FFT_SIZE, true);
    let mut prev_input = Complex::new(0.0, 0.0); // sample n-1
    let mut prev_diff = Complex::new(0.0, 0.0); // sample n-1 - sample n-2

    //for (x,t) in time_sample[..].iter()
    //    .zip(data_complex_buffer[..].iter_mut()) {
    //        *t = Complex::new(gain * x, 0.0);
    //}

    for i in 0..FFT_SIZE {
        let mut re_sum: f32 = 0.0;
        for j in 0..7 {
            re_sum += (2.0 * std::f64::consts::PI * (i as f64 / freq * (j as f64))).sin() as f32;
        }
        data_complex_buffer[i].re = re_sum;
    }

    // println!("generate_sample {} {} {} {} {}, length {}",
    //     data_complex_buffer[0].re,
    //     data_complex_buffer[63].re, 
    //     data_complex_buffer[127].re,
    //     data_complex_buffer[191].re,
    //     data_complex_buffer[255].re, 
    //     data_complex_buffer.len());

    fft.process(&data_complex_buffer[..], &mut complex_freq_buffer[..]);

    for (x, y) in analytic_filter.iter().zip(complex_freq_buffer.iter_mut()) {
        *y = *x * *y;
    }

    ifft.process(&complex_freq_buffer[..], &mut complex_analytic_buffer[..]);

    // let mut count: i32 = 0;
    // for(x) in complex_analytic_buffer.iter() {
    //     println!("{} {}", count, x);
    //     count += 1;

    //     if(count > 100){ break};
    // }

    analytic_buffer[0] = analytic_buffer[BUFFER_SIZE];
    analytic_buffer[1] = analytic_buffer[BUFFER_SIZE + 1];
    analytic_buffer[2] = analytic_buffer[BUFFER_SIZE + 2];
    let scale = FFT_SIZE as f32;
    for (&x, y) in complex_analytic_buffer[FFT_SIZE - BUFFER_SIZE..].iter()
        .zip(analytic_buffer[3..].iter_mut()) {

        let diff = x - prev_input; // vector
        prev_input = x;

        let angle = get_angle(diff, prev_diff).abs().log2().max(-1.0e12); // angular velocity (with processing)
        prev_diff = diff;

        *y = Vec4 { vec: [
            x.re / scale,
            x.im / scale,
            0.75, //angle,
            0f32,
        ]};
    }

    let mut buffer = buffers[buffer_index].lock().unwrap();
    buffer.analytic.copy_from_slice(&analytic_buffer[..]);
    buffer.rendered = false;
 
    // let mut count: i32 = 0;
    // for(x) in analytic_buffer.iter() {
    //     println!("{} {} {}", count, x.vec[0], x.vec[1]);
    //     count += 1;
    // }

    buffer_index = (buffer_index + 1) % NUM_BUFFERS;
}

// angle between two complex numbers
// scales into [0, 0.5]
pub fn get_angle(v: Complex<f32>, u: Complex<f32>) -> f32 {
    // 2 atan2(len(len(v)*u − len(u)*v), len(len(v)*u + len(u)*v))
    let len_v_mul_u = v.norm_sqr().sqrt() * u;
    let len_u_mul_v = u.norm_sqr().sqrt() * v;
    let left = (len_v_mul_u - len_u_mul_v).norm_sqr().sqrt(); // this is positive
    let right = (len_v_mul_u + len_u_mul_v).norm_sqr().sqrt(); // this is positive
    left.atan2(right) / ::std::f32::consts::PI
}

// returns biquad lowpass filter
fn get_lowpass(n: f32, q: f32) -> Box<FnMut(f32) -> f32> {
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

    println!("make_analytic, n={}, m={}.", n, m);

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
