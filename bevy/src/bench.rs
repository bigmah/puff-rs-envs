//! How fast a Bevy world steps, with random actions and no trainer (puffer_bevy::bench).
//!
//!     cargo +1.99 run --release -p bevy-envs --bin bevy-bench [-- --over-time]

use bevy_envs::pusher::Pusher;
use bevy_envs::so101::So101;
use bevy_envs::so101_3d::So101x3;

fn main() {
    puffer_bevy::bench::all("pusher", Pusher::new, 200);
    puffer_bevy::bench::all("so101", So101::new, 500);
    puffer_bevy::three::bench::all("so101_3d", So101x3::new, 500);
}
