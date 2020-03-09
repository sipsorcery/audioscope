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
    let audio_context = init_audio(&config);
    let mut planner = periodic::Planner::new();
    let shared_buffer = audio_context.displayBuffers.clone();
    let mut curr_freq = 440.0;
    //generate_sample(&shared_buffer, curr_freq);
    planner.add(
        move ||  {
            let mut curr_freq: f64 = 25.0 as f64;
            // if time::Time::now().second() % 2 == 0 {
            //     curr_freq = 880.0;
            // }
            generate_sample(&shared_buffer, curr_freq);
        },
        //periodic::Every::new(Duration::from_millis(30)),
        periodic::Every::new(Duration::from_secs(3)),
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
    display(&config, &audio_context.displayBuffers);

    // let mut stdout = stdout();
    // stdout.write(b"Press Enter to continue...").unwrap();
    // stdout.flush().unwrap();
    // stdin().read(&mut [0]).unwrap();
}
