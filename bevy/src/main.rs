//! pufferl for the Bevy envs: each is many arenas in a headless Bevy world (lib.rs).
//!
//!     cargo +1.99 run --release -p puffer-bevy --bin bevy-envs -- train bevy_cartpole
//!     cargo +1.99 run --release -p puffer-bevy --bin bevy-envs -- train bevy_pusher

use puffer_bevy::{cartpole::CartPole, pusher::Pusher, Arenas};
use pufferl::{cli, Envs};

fn main() {
    cli::main(
        Envs::new()
            .add("bevy_cartpole", include_str!("../config/bevy_cartpole.ini"), |config, i| Arenas::from_config(CartPole::new(config), config, i))
            .add("bevy_pusher", include_str!("../config/bevy_pusher.ini"), |config, i| Arenas::from_config(Pusher::new(config), config, i)),
    );
}
