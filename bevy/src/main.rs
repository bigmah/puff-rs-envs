//! pufferl for the Bevy envs: each is many arenas in a headless Bevy world (lib.rs).
//!
//!     cargo +1.99 run --release -p puffer-bevy --bin bevy-envs -- train bevy_pusher
//!     cargo +1.99 run --release -p puffer-bevy --bin bevy-envs -- eval bevy_pusher latest
//!
//! eval without --headless opens a window and watches [env] num_agents arenas play
//! (view.rs): the window has the main thread, and eval runs behind it.

use puffer_bevy::{cartpole::CartPole, pusher::Pusher, view, Arenas};
use pufferl::{cli, Envs};

fn envs() -> Envs {
    Envs::new()
        .add("bevy_cartpole", include_str!("../config/bevy_cartpole.ini"), |config, i| Arenas::from_config(CartPole::new(config), config, i))
        .add("bevy_pusher", include_str!("../config/bevy_pusher.ini"), |config, i| Arenas::from_config(Pusher::new(config), config, i))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let watch = args.get(1).is_some_and(|mode| mode == "eval") && !args.iter().any(|arg| arg == "--headless");
    if watch {
        view::run(|| cli::main(envs()));
    } else {
        cli::main(envs());
    }
}
