//! How fast a Bevy world steps, with random actions and no trainer (puffer_bevy::bench).
//!
//!     cargo +1.99 run --release -p bevy-envs --bin bevy-bench [-- --over-time]

use bevy_envs::pusher::Pusher;

fn main() {
    puffer_bevy::bench::all("pusher", Pusher::new, 200);
}
