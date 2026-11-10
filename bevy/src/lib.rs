//! Envs whose physics is Bevy and Avian, on puff-rs-bevy: each is a [`BevyEnv`], one arena
//! and its rules, which puff-rs-bevy's `Arenas` runs many to a headless world.
//!
//! [`BevyEnv`]: puffer_bevy::BevyEnv

pub mod pusher;
pub mod so101;
pub mod so101_3d;

#[cfg(test)]
mod tests {
    use crate::pusher::Pusher;
    use puffer_bevy::bench::{config, Buffers};
    use puffer_bevy::bevy::math::Vec2;
    use puffer_bevy::Arenas;
    use pufferl::env::Env;

    /// Step every arena with the action policy gives its observation, n times; the
    /// observations after, and the reward over them all
    fn run(env: &mut Arenas<Pusher>, n: usize, policy: impl Fn(&[f32]) -> f32) -> (Vec<f32>, f32) {
        let mut buffers = Buffers::new(env);
        env.reset(buffers.agents());
        let mut reward = 0.0;
        for _ in 0..n {
            let obs_size = buffers.obs_size;
            for (action, obs) in buffers.actions.iter_mut().zip(buffers.obs.chunks(obs_size)) {
                *action = policy(obs);
            }
            env.step(buffers.agents());
            reward += buffers.rewards.iter().sum::<f32>();
        }
        (buffers.obs, reward)
    }

    #[test]
    fn agent_moves_and_walls_hold() {
        let mut env = Arenas::new(Pusher::new(&config(4, 1000)), 4, 0);
        // Right for a long time: the agent ends at the right wall, not through it
        let (obs, _) = run(&mut env, 300, |_| 4.0);
        for room in obs.chunks(10) {
            let x = room[0] * 4.0;
            assert!(x > 3.0 && x < 4.0, "agent at x {x}, the wall's inside is at 4");
        }
    }

    #[test]
    fn box_moves_when_pushed() {
        // Walk at the box: it moves from where it started
        let config = config(8, 1000);
        let mut env = Arenas::new(Pusher::new(&config), 8, 3);
        let (start, _) = run(&mut env, 0, |_| 0.0);
        let (end, _) = run(&mut env, 200, |obs| {
            let to_box = (obs[4] - obs[0], obs[5] - obs[1]);
            (if to_box.0.abs() > to_box.1.abs() { if to_box.0 > 0.0 { 4 } else { 3 } } else if to_box.1 > 0.0 { 1 } else { 2 }) as f32
        });
        let moved = start.chunks(10).zip(end.chunks(10)).filter(|(a, b)| (a[4] - b[4]).abs() + (a[5] - b[5]).abs() > 0.05).count();
        assert!(moved >= 6, "{moved} of 8 boxes moved");
    }

    #[test]
    fn scripted_solves_some() {
        let mut env = Arenas::new(Pusher::new(&config(32, 200)), 32, 5);
        let (_, total) = run(&mut env, 1000, |o| {
            let (agent, block, goal) = (Vec2::new(o[0], o[1]), Vec2::new(o[4], o[5]), Vec2::new(o[8], o[9]));
            // Get behind the box, then push it at the goal
            let behind = block - (goal - block).normalize_or_zero() * 0.25;
            let to = if agent.distance(behind) > 0.08 { behind - agent } else { goal - agent };
            (if to.x.abs() > to.y.abs() { if to.x > 0.0 { 4 } else { 3 } } else if to.y > 0.0 { 1 } else { 2 }) as f32
        });
        let log = env.log();
        let solved = log.sums.iter().find(|(k, _)| *k == "perf").map_or(0.0, |(_, v)| *v);
        println!("episodes {} solved {solved} reward {total}", log.n);
        assert!(solved > 0.0);
    }
}
