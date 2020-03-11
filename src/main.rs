extern crate rustc_serialize;
extern crate toml;
#[macro_use]
extern crate glium;
extern crate num;
extern crate portaudio;
extern crate rustfft;
extern crate time;

mod audio;
mod config;
mod display;
mod file_loader;

use audio::{
    get_lowpass, get_sample, init_audio, init_portaudio, process_sample
};
use config::load_config;
use display::display;
use std::time::{Duration, SystemTime};
use std::{
    ops::DerefMut,
    sync::{Arc, Mutex},
};

fn main() {
    let config = load_config();
    let buffer_size = config.audio.buffer_size as usize;
    let cutoff = config.audio.cutoff;
    let q = config.audio.q;

    let display_config = load_config();

    let (mut audio_context, display_buffer) = init_audio(&config);
    
    let confg_arc = Arc::new(Mutex::new(config));

    let mut stream = init_portaudio(&confg_arc, &mut audio_context, &mut display_buffer.clone()).unwrap();
    stream.start().unwrap();
    display(&display_config, display_buffer);
    stream.stop().unwrap(); 

    /* let audio_mutex = std::sync::Mutex::new(audio_context);
    let audio_arc = std::sync::Arc::new(audio_mutex);
    let display_buffer_producer = display_buffer.clone();

    let angle_lp = get_lowpass(cutoff, q);
    let angle_lp_mutex = Mutex::new(angle_lp);
    let angle_lp_arc = Arc::new(angle_lp_mutex);

    let noise_lp = get_lowpass(0.05, 0.7);
    let noise_lp_mutex = Mutex::new(noise_lp);
    let noise_lp_arc = Arc::new(noise_lp_mutex);

    let base_freq = 440.0;
    let start_time = std::time::Instant::now();
    let mut planner = periodic::Planner::new();

    planner.add(
         move ||  {
            let freq = base_freq;// + start_time.elapsed().as_secs() as f32 * 100.0;
            let mut ctx =  audio_arc.lock().unwrap();
            let mut angle_lp_fn = angle_lp_arc.lock().unwrap();
            let mut noise_lp_fn = noise_lp_arc.lock().unwrap();

            let audio_sample = get_sample(freq, buffer_size);

            process_sample(&config, &mut ctx, &audio_sample, &display_buffer_producer, angle_lp_fn.deref_mut(), noise_lp_fn.deref_mut());
        },
        //periodic::Every::new(Duration::from_millis(30)),
        periodic::Every::new(Duration::from_secs(1)),
    );
    planner.start(); 

    display(&display_config, display_buffer); */

    // let mut stdout = stdout();
    // stdout.write(b"Press Enter to continue...").unwrap();
    // stdout.flush().unwrap();
    // stdin().read(&mut [0]).unwrap();

    //std::process::exit(0);
}
