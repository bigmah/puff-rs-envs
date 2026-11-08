//! pufferl for the Bevy envs: each is many arenas in a headless Bevy world (puff-rs-bevy).
//!
//!     cargo +1.99 run --release -p bevy-envs --bin bevy-envs -- train bevy_pusher
//!     cargo +1.99 run --release -p bevy-envs --bin bevy-envs -- eval bevy_pusher latest
//!
//! eval without --headless opens a window and watches [env] num_agents arenas play: the
//! window has the main thread, and eval runs behind it.

use bevy_envs::pusher::Pusher;
use puffer_bevy::Arenas;
use pufferl::{cli, Envs};

fn envs() -> Envs {
    Envs::new().add("bevy_pusher", include_str!("../config/bevy_pusher.ini"), |config, i| Arenas::from_config(Pusher::new(config), config, i))
}

fn main() {
    puffer_bevy::run(|| cli::main(envs()));
}
