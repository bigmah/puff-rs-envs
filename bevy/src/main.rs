//! pufferl for the Bevy envs: each is many arenas in a headless Bevy world (puff-rs-bevy).
//!
//!     cargo +1.99 run --release -p bevy-envs --bin bevy-envs -- train bevy_pusher
//!     cargo +1.99 run --release -p bevy-envs --bin bevy-envs -- eval bevy_pusher latest
//!
//! eval without --headless opens a window and watches [env] num_agents arenas play: the
//! window has the main thread, and eval runs behind it. 3D envs get a 3D window.

use bevy_envs::pusher::Pusher;
use bevy_envs::so101::So101;
use bevy_envs::so101_3d::So101x3;
use puffer_bevy::three::Arenas3d;
use puffer_bevy::Arenas;
use pufferl::{cli, Envs};

fn envs() -> Envs {
    Envs::new()
        .add("bevy_pusher", include_str!("../config/bevy_pusher.ini"), |config, i| Arenas::from_config(Pusher::new(config), config, i))
        .add("bevy_so101", include_str!("../config/bevy_so101.ini"), |config, i| Arenas::from_config(So101::new(config), config, i))
        .add("bevy_so101_3d", include_str!("../config/bevy_so101_3d.ini"), |config, i| Arenas3d::from_config(So101x3::new(config), config, i))
}

/// The envs drawn in three dimensions
const THREE_D: &[&str] = &["bevy_so101_3d"];

fn main() {
    let three_d = std::env::args().nth(2).is_some_and(|env| THREE_D.contains(&env.as_str()));
    if three_d {
        puffer_bevy::three::run(|| cli::main(envs()));
    } else {
        puffer_bevy::run(|| cli::main(envs()));
    }
}
