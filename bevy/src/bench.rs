//! How fast a Bevy world steps, with random actions and no trainer: one world on one
//! thread at a few arena counts, then a world per thread on every core.
//!
//!     cargo run --release -p puffer-bevy --bin bevy-bench

use std::time::Instant;

use puffer_bevy::{cartpole::CartPole, pusher::Pusher, Arenas, BevyEnv, Rng};
use pufferl::env::{Agents, Config, Env};

fn config(arenas: usize, max_steps: usize) -> Config {
    Config::new(vec![("arenas".into(), arenas.to_string()), ("max_steps".into(), max_steps.to_string())])
}

/// The trainer's buffers for one env
struct Buffers {
    obs: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    terminals: Vec<f32>,
    mask: Vec<u8>,
    obs_size: usize,
    heads: Vec<usize>,
}

impl Buffers {
    fn new<E: Env<Obs = f32>>(env: &E) -> Self {
        let n = env.num_agents();
        let heads = env.act_sizes();
        Self {
            obs: vec![0.0; n * env.obs_size()],
            actions: vec![0.0; n * heads.len()],
            rewards: vec![0.0; n],
            terminals: vec![0.0; n],
            mask: vec![1; n * heads.iter().sum::<usize>()],
            obs_size: env.obs_size(),
            heads,
        }
    }

    fn agents(&mut self) -> Agents<'_, f32> {
        self.rewards.fill(0.0);
        self.terminals.fill(0.0);
        Agents {
            observations: &mut self.obs,
            actions: &self.actions,
            rewards: &mut self.rewards,
            terminals: &mut self.terminals,
            action_mask: &mut self.mask,
            obs_size: self.obs_size,
            num_atns: self.heads.len(),
            mask_size: self.heads.iter().sum(),
        }
    }
}

/// Agent steps a second of env with random actions, over about seconds of stepping
fn steps_per_second<E: Env<Obs = f32>>(env: &mut E, seconds: f64) -> f64 {
    let mut buffers = Buffers::new(env);
    let mut rng = Rng::new(7);
    env.reset(buffers.agents());
    let (start, mut steps) = (Instant::now(), 0usize);
    while start.elapsed().as_secs_f64() < seconds {
        let heads = buffers.heads.clone();
        for (a, head) in buffers.actions.iter_mut().zip(heads.iter().cycle()) {
            *a = (rng.next_u64() % *head as u64) as f32;
        }
        env.step(buffers.agents());
        steps += env.num_agents();
    }
    steps as f64 / start.elapsed().as_secs_f64()
}

fn bench<E: BevyEnv>(name: &str, make: fn(&Config) -> E, max_steps: usize) {
    println!("{name}, one world on one thread:");
    for arenas in [1, 8, 64, 256, 512, 1024] {
        let config = config(arenas, max_steps);
        let mut env = Arenas::new(make(&config), arenas, 0);
        let sps = steps_per_second(&mut env, 1.0);
        println!("  {arenas:5} arenas   {sps:>10.0} steps/s   {:>7.2} us per arena step", 1e6 / sps);
    }
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    for workers in [8, threads] {
        let arenas = 512;
        let handles: Vec<_> = (0..workers)
            .map(|t| std::thread::spawn(move || {
                let config = config(arenas, max_steps);
                let mut env = Arenas::new(make(&config), arenas, t);
                steps_per_second(&mut env, 3.0)
            }))
            .collect();
        let total: f64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        println!("  {workers:2} threads x {arenas} arenas: {total:>10.0} steps/s");
    }
}

fn respawn_cost<E: BevyEnv>(name: &str, make: fn(&Config) -> E) {
    let config = config(512, 500);
    let mut env = Arenas::new(make(&config), 512, 0);
    let mut buffers = Buffers::new(&env);
    env.reset(buffers.agents());
    let start = Instant::now();
    for _ in 0..20 {
        env.reset(buffers.agents());
    }
    println!("{name}: respawning an arena takes {:.1} us", start.elapsed().as_secs_f64() * 1e6 / (20.0 * 512.0));
}

fn over_time<E: BevyEnv>(name: &str, make: fn(&Config) -> E, max_steps: usize) {
    let config = config(512, max_steps);
    let mut env = Arenas::new(make(&config), 512, 0);
    print!("{name}, 512 arenas, steps/s in 3 s windows:");
    for _ in 0..10 {
        print!(" {:.0}K", steps_per_second(&mut env, 3.0) / 1e3);
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }
    println!();
}

fn main() {
    if std::env::args().any(|a| a == "--over-time") {
        over_time("cartpole", CartPole::new, 500);
        over_time("pusher", Pusher::new, 50);
        return;
    }
    respawn_cost("cartpole", CartPole::new);
    respawn_cost("pusher", Pusher::new);
    bench("cartpole", CartPole::new, 500);
    bench("pusher", Pusher::new, 200);
}
