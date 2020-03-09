extern crate toml;
extern crate rustc_serialize;
#[macro_use]
extern crate glium;
extern crate portaudio;
extern crate num;
extern crate rustfft;
extern crate time;

mod file_loader;
mod config;
mod audio;
mod display;

use num::complex::Complex;
use config::load_config;
use audio::init_audio;
use audio::generate_sample;
use display::display;
use std::time::{Duration, SystemTime};
use std::io::{stdin, stdout, Read, Write};

fn main() {
    let config = load_config();
    let (audio_context, display_buffer) = init_audio(&config);
    let audio_mutex = std::sync::Mutex::new(audio_context);
    
    let audio_arc = std::sync::Arc::new(audio_mutex);
    let display_buffer_producer = display_buffer.clone();
    //let display_arc = audio_arc.clone();

    let base_freq = 440.0;
    let start_time = std::time::Instant::now();
    let mut planner = periodic::Planner::new();

    //let shared_buffer = audio_context.lock().unwrap().displayBuffers.clone();
    //generate_sample(&shared_buffer, curr_freq);
    planner.add(
         move ||  {
            let freq = base_freq + start_time.elapsed().as_secs() as f32 * 100.0;
            let mut ctx =  audio_arc.lock().unwrap();

            generate_sample(&mut ctx, &display_buffer_producer, freq);
        },
        //periodic::Every::new(Duration::from_millis(30)),
        periodic::Every::new(Duration::from_millis(30)),
    );
    planner.start();
    // let audio_mutex = std::sync::Mutex::new(audio_context);
    // let shared_audio = std::sync::Arc::new(audio_mutex);
    // let child;
    // {
    //     let audio_arc = shared_audio.clone();
    //     child = std::thread::spawn(move || {
    //         let mut thd_freq = 440.0;
    //         thd_freq = generate_sample(audio_arc, thd_freq);
    //     });
    // }


    display(&config, &display_buffer);

    // let mut stdout = stdout();
    // stdout.write(b"Press Enter to continue...").unwrap();
    // stdout.flush().unwrap();
    // stdin().read(&mut [0]).unwrap();

    //std::process::exit(0);
}
