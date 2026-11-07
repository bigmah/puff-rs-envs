//! Envs whose physics is Bevy and Avian, for puff-rs.
//!
//! An env is a [`BevyEnv`]: how to spawn one arena (its bodies, colliders and joints), how
//! an action pushes on it, what its agent sees, and what a step pays. [`Arenas`] puts many
//! arenas side by side in one headless Bevy world, far enough apart that they never touch,
//! and is one pufferl [`Env`] with an agent per arena: a step applies every arena's action,
//! advances the world one physics step, then reads every arena back. The vecenv makes as
//! many Arenas as `vec.total_agents / env.arenas` and steps them on its threads, so each
//! world runs single-threaded and the machine's cores go to worlds, not to one world.
//!
//! An arena's bodies are spawned once. An episode starts by putting them back (reset), not
//! by despawning and spawning them again: a world that despawns and spawns steadily gets
//! slower the longer it runs, by half in half a minute.

use std::time::{Duration, Instant};

use avian2d::prelude::*;
use bevy::app::{App, Main, TaskPoolPlugin};
use bevy::ecs::entity::Entity;
use bevy::ecs::component::{Component, Mutable};
use bevy::ecs::world::{EntityWorldMut, World};
use bevy::math::{Quat, Vec2, Vec3};
use bevy::time::{Fixed, Time, TimePlugin, TimeUpdateStrategy};
use bevy::transform::components::Transform;
use bevy::transform::TransformPlugin;
use pufferl::env::{Agents, Config, Env, Log};

pub mod cartpole;
pub mod pusher;
pub mod view;

pub use avian2d;
pub use bevy;

/// What an env writes: one arena, and the rules for it. Arena is the env's own handle on
/// a spawned arena, the entities it acts on and reads, say.
pub trait BevyEnv: Send + 'static {
    type Arena: Send;

    fn obs_size(&self) -> usize;

    /// How many actions each head has
    fn act_sizes(&self) -> Vec<usize>;

    /// Seconds of physics in a step
    fn dt(&self) -> f32 {
        0.02
    }

    /// Room for one arena: arenas are this far apart, center to center
    fn spacing(&self) -> f32 {
        20.0
    }

    /// Anything the world needs past physics: gravity, a resource, systems of its own
    fn build(&self, _app: &mut App) {}

    /// One arena's bodies, colliders and joints, around origin, once: where they start an
    /// episode is reset's to say
    fn spawn(&self, world: &mut World, origin: Vec2) -> Self::Arena;

    /// Start an episode: put the arena's bodies where an episode starts, with [`teleport`]
    fn reset(&self, world: &mut World, arena: &mut Self::Arena, origin: Vec2, rng: &mut Rng);

    /// Act on the arena: actions holds one action per head
    fn act(&self, world: &mut World, arena: &mut Self::Arena, actions: &[usize]);

    /// What the arena's agent sees
    fn observe(&self, world: &World, arena: &Self::Arena, origin: Vec2, obs: &mut [f32]);

    /// The reward for the step just taken, the tick-th of the episode, and whether the
    /// episode is over
    fn outcome(&self, world: &World, arena: &mut Self::Arena, origin: Vec2, tick: usize) -> (f32, bool);

    /// What to draw past the bodies, which are drawn from their colliders: a goal, a track
    fn marks(&self, _world: &World, _arena: &Self::Arena, _origin: Vec2, _marks: &mut Vec<view::Mark>) {}

    /// An ended episode into the log: by default its return as the score, and its length
    fn log_episode(&self, log: &mut Log, _arena: &Self::Arena, episode_return: f64, length: usize) {
        log.episode(&[("score", episode_return), ("episode_return", episode_return), ("episode_length", length as f64)]);
    }
}

/// A body's place, as both Bevy (Transform) and Avian (Position, Rotation) have it, so
/// neither syncs the other back to the origin on its first step
pub fn placed(at: Vec2, angle: f32) -> (Transform, Position, Rotation) {
    (Transform::from_translation(Vec3::new(at.x, at.y, 0.0)).with_rotation(Quat::from_rotation_z(angle)), Position(at), Rotation::radians(angle))
}

/// Move a body to at, turned by angle and moving at velocity and spin, as a new episode
/// starts it. Avian wakes a body whose place or velocity is set.
pub fn teleport(world: &mut World, entity: Entity, at: Vec2, angle: f32, velocity: Vec2, spin: f32) {
    let (transform, position, rotation) = placed(at, angle);
    let mut body = world.entity_mut(entity);
    // Written in place rather than inserted, so Avian sees a change and not a new body
    set(&mut body, transform);
    set(&mut body, position);
    set(&mut body, rotation);
    set(&mut body, LinearVelocity(velocity));
    set(&mut body, AngularVelocity(spin));
}

fn set<C: Component<Mutability = Mutable>>(body: &mut EntityWorldMut, value: C) {
    match body.get_mut::<C>() {
        Some(mut component) => *component = value,
        None => {
            body.insert(value);
        }
    }
}

/// splitmix64
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi)
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
}

struct Slot<A> {
    arena: A,
    origin: Vec2,
    tick: usize,
    episode_return: f64,
}

/// Many arenas of one env in one world, as one pufferl env with an agent per arena
pub struct Arenas<E: BevyEnv> {
    env: E,
    world: World,
    slots: Vec<Slot<E::Arena>>,
    rng: Rng,
    log: Log,
    heads: Vec<usize>,
    /// How many times real time a watched world runs at
    view_speed: f32,
    last_render: Option<Instant>,
}

impl<E: BevyEnv> Arenas<E> {
    /// count arenas of env, in a world of their own. index seeds the arenas' randomness.
    pub fn new(env: E, count: usize, index: usize) -> Self {
        if count == 0 {
            pufferl::env::fail("[env] arenas should be at least 1");
        }
        let mut app = App::new();
        app.add_plugins((TaskPoolPlugin::default(), TimePlugin, TransformPlugin, PhysicsPlugins::default()))
            // Each update runs the fixed loop, and so physics, exactly once
            .insert_resource(Time::<Fixed>::from_duration(Duration::from_secs_f32(env.dt())))
            .insert_resource(TimeUpdateStrategy::FixedTimesteps(1));
        env.build(&mut app);
        app.finish();
        app.cleanup();
        // The App holds a runner that may not cross threads, and the world, schedules and
        // all, may: the world is what steps
        let world = std::mem::take(app.world_mut());

        let heads = vec![0; env.act_sizes().len()];
        let side = (count as f64).sqrt().ceil() as usize;
        let spacing = env.spacing();
        let mut arenas = Self { env, world, slots: Vec::with_capacity(count), rng: Rng::new(index as u64), log: Log::default(), heads, view_speed: 1.0, last_render: None };
        for i in 0..count {
            let origin = Vec2::new((i % side) as f32, (i / side) as f32) * spacing;
            let mut arena = arenas.env.spawn(&mut arenas.world, origin);
            arenas.env.reset(&mut arenas.world, &mut arena, origin, &mut arenas.rng);
            arenas.slots.push(Slot { arena, origin, tick: 0, episode_return: 0.0 });
        }
        arenas
    }

    /// [env] arenas of env, or num_agents when a window is watching: eval with a window
    /// makes one env of num_agents agents, so those are what is on screen. [env]
    /// view_speed is how many times real time it runs at.
    pub fn from_config(env: E, config: &Config, index: usize) -> Self {
        let count = if view::watching() { config.size("num_agents") } else { config.size("arenas") };
        let mut arenas = Self::new(env, count, index);
        arenas.view_speed = config.num_or("view_speed", 1.0) as f32;
        arenas
    }

    /// A new episode in arena i
    fn restart(&mut self, i: usize) {
        let slot = &mut self.slots[i];
        self.env.reset(&mut self.world, &mut slot.arena, slot.origin, &mut self.rng);
        (slot.tick, slot.episode_return) = (0, 0.0);
    }

    /// One physics step of the whole world, as App::update would
    pub fn update(&mut self) {
        self.world.run_schedule(Main);
        self.world.clear_trackers();
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn env(&self) -> &E {
        &self.env
    }
}

impl<E: BevyEnv> Env for Arenas<E> {
    type Obs = f32;

    fn num_agents(&self) -> usize {
        self.slots.len()
    }

    fn obs_size(&self) -> usize {
        self.env.obs_size()
    }

    fn act_sizes(&self) -> Vec<usize> {
        self.env.act_sizes()
    }

    fn reset(&mut self, mut agents: Agents<'_, f32>) {
        for i in 0..self.slots.len() {
            self.restart(i);
            let slot = &self.slots[i];
            self.env.observe(&self.world, &slot.arena, slot.origin, agents.observation(i));
        }
    }

    fn step(&mut self, mut agents: Agents<'_, f32>) {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            for (head, action) in self.heads.iter_mut().enumerate() {
                *action = agents.action(i, head);
            }
            self.env.act(&mut self.world, &mut slot.arena, &self.heads);
        }
        self.update();
        for i in 0..self.slots.len() {
            let slot = &mut self.slots[i];
            slot.tick += 1;
            let (reward, done) = self.env.outcome(&self.world, &mut slot.arena, slot.origin, slot.tick);
            slot.episode_return += reward as f64;
            agents.rewards[i] = reward;
            if done {
                agents.terminals[i] = 1.0;
                self.env.log_episode(&mut self.log, &slot.arena, slot.episode_return, slot.tick);
                self.restart(i);
            }
            let slot = &self.slots[i];
            self.env.observe(&self.world, &slot.arena, slot.origin, agents.observation(i));
        }
    }

    fn log(&mut self) -> &mut Log {
        &mut self.log
    }

    /// Every body in the world as its collider has it, for the window, at real time
    fn render(&mut self) {
        if !view::watching() {
            return;
        }
        let step = Duration::from_secs_f32(self.env.dt() / self.view_speed.max(0.01));
        if let Some(last) = self.last_render {
            std::thread::sleep(step.saturating_sub(last.elapsed()));
        }
        self.last_render = Some(Instant::now());

        let mut bodies: Vec<(Entity, view::Body)> = Vec::new();
        let mut query = self.world.query::<(Entity, &Collider, &Position, &Rotation, &RigidBody)>();
        for (entity, collider, position, rotation, body) in query.iter(&self.world) {
            let shape = collider.shape_scaled();
            let outline = if let Some(ball) = shape.as_ball() {
                view::Outline::Circle(ball.radius)
            } else if let Some(cuboid) = shape.as_cuboid() {
                view::Outline::Rect(Vec2::new(cuboid.half_extents.x, cuboid.half_extents.y))
            } else {
                let aabb = shape.compute_local_aabb();
                view::Outline::Rect(Vec2::new(aabb.half_extents().x, aabb.half_extents().y))
            };
            bodies.push((entity, view::Body { at: position.0, angle: rotation.as_radians(), outline, dynamic: body.is_dynamic() }));
        }
        // In spawn order: a body that sleeps moves to another table, and the window matches
        // bodies to what it drew by place in the list
        bodies.sort_by_key(|(entity, _)| *entity);
        let mut marks = Vec::new();
        for slot in &self.slots {
            self.env.marks(&self.world, &slot.arena, slot.origin, &mut marks);
        }
        view::show(view::Frame { bodies: bodies.into_iter().map(|(_, body)| body).collect(), marks });
    }

    fn window_closed(&self) -> bool {
        view::closed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartpole::CartPole;
    use crate::pusher::Pusher;

    /// Step every arena with the action action gives it, n times; the observations after
    fn run<E: BevyEnv>(env: &mut Arenas<E>, n: usize, mut action: impl FnMut(&[f32]) -> f32) -> Vec<f32> {
        let agents = env.num_agents();
        let (obs_size, heads) = (env.obs_size(), env.act_sizes().len());
        let mut obs = vec![0.0; agents * obs_size];
        let (mut r, mut t, mut m) = (vec![0.0; agents], vec![0.0; agents], vec![1u8; agents * 16]);
        let mut actions = vec![0.0; agents * heads];
        macro_rules! mk {
            () => {
                Agents { observations: &mut obs, actions: &actions, rewards: &mut r, terminals: &mut t, action_mask: &mut m, obs_size, num_atns: heads, mask_size: 2 }
            };
        }
        env.reset(mk!());
        for _ in 0..n {
            for i in 0..agents {
                actions[i * heads] = action(&obs[i * obs_size..(i + 1) * obs_size]);
            }
            r.fill(0.0);
            t.fill(0.0);
            env.step(mk!());
        }
        obs
    }

    fn config(arenas: usize, max_steps: usize) -> Config {
        Config::new(vec![("arenas".into(), arenas.to_string()), ("max_steps".into(), max_steps.to_string())])
    }

    fn mean(log: &Log, key: &str) -> f64 {
        log.sums.iter().find(|(k, _)| *k == key).map_or(f64::NAN, |(_, v)| v / log.n)
    }

    #[test]
    fn cartpole_falls_when_pushed_one_way_and_stands_when_balanced() {
        let config = config(16, 500);
        // Pushing right every step: every arena falls, well inside a hundred steps
        let mut env = Arenas::new(CartPole::new(&config), 16, 0);
        run(&mut env, 100, |_| 1.0);
        assert!(env.log.n >= 16.0, "{} episodes", env.log.n);
        let length = mean(&env.log, "episode_length");
        assert!(length > 5.0 && length < 100.0, "pushed one way, episodes last {length}");

        // Toward the lean and the swing, and back from the track's ends: every arena lasts the 500
        let mut env = Arenas::new(CartPole::new(&config), 16, 1);
        run(&mut env, 500, |obs| if obs[2] + 2.0 * obs[3] + 0.3 * obs[0] + 0.5 * obs[1] > 0.0 { 1.0 } else { 0.0 });
        assert_eq!(env.log.n, 16.0, "balanced: {} episodes, mean length {}", env.log.n, mean(&env.log, "episode_length"));
        assert_eq!(mean(&env.log, "episode_length"), 500.0);
    }

    #[test]
    fn pusher_agent_moves_and_walls_hold() {
        let config = config(4, 1000);
        let mut env = Arenas::new(Pusher::new(&config), 4, 0);
        // Right for a long time: the agent ends at the right wall, not through it
        let obs = run(&mut env, 300, |_| 4.0);
        for room in obs.chunks(10) {
            let x = room[0] * 4.0;
            assert!(x > 3.0 && x < 4.0, "agent at x {x}, the wall's inside is at 4");
        }
    }

    #[test]
    fn pusher_box_moves_when_pushed() {
        // Walk at the box: it moves from where it started
        let config = config(8, 1000);
        let mut env = Arenas::new(Pusher::new(&config), 8, 3);
        let start = run(&mut env, 0, |_| 0.0);
        let end = run(&mut env, 200, |obs| {
            let to_box = (obs[4] - obs[0], obs[5] - obs[1]);
            (if to_box.0.abs() > to_box.1.abs() { if to_box.0 > 0.0 { 4 } else { 3 } } else if to_box.1 > 0.0 { 1 } else { 2 }) as f32
        });
        let moved = start.chunks(10).zip(end.chunks(10)).filter(|(a, b)| (a[4] - b[4]).abs() + (a[5] - b[5]).abs() > 0.05).count();
        assert!(moved >= 6, "{moved} of 8 boxes moved");
    }
}

#[cfg(test)]
mod solvable {
    use super::*;
    use crate::pusher::Pusher;

    #[test]
    fn pusher_scripted_solves_some() {
        let config = Config::new(vec![("arenas".into(), "32".into()), ("max_steps".into(), "200".into())]);
        let mut env = Arenas::new(Pusher::new(&config), 32, 5);
        let (n, obs_size) = (32, 10);
        let (mut obs, mut actions) = (vec![0.0; n * obs_size], vec![0.0; n]);
        let (mut r, mut t, mut m) = (vec![0.0; n], vec![0.0; n], vec![1u8; n * 5]);
        macro_rules! agents {
            () => {
                Agents { observations: &mut obs, actions: &actions, rewards: &mut r, terminals: &mut t, action_mask: &mut m, obs_size, num_atns: 1, mask_size: 5 }
            };
        }
        env.reset(agents!());
        let mut total = 0.0;
        for _ in 0..1000 {
            for i in 0..n {
                let o = &obs[i * obs_size..(i + 1) * obs_size];
                let (agent, block, goal) = (Vec2::new(o[0], o[1]), Vec2::new(o[4], o[5]), Vec2::new(o[8], o[9]));
                // Get behind the box, then push it at the goal
                let behind = block - (goal - block).normalize_or_zero() * 0.25;
                let to = if agent.distance(behind) > 0.08 { behind - agent } else { goal - agent };
                actions[i] = if to.x.abs() > to.y.abs() { if to.x > 0.0 { 4.0 } else { 3.0 } } else if to.y > 0.0 { 1.0 } else { 2.0 };
            }
            r.fill(0.0);
            t.fill(0.0);
            env.step(agents!());
            total += r.iter().sum::<f32>();
        }
        let solved = env.log.sums.iter().find(|(k, _)| *k == "perf").map_or(0.0, |(_, v)| *v);
        println!("episodes {} solved {solved} reward {total}", env.log.n);
        assert!(solved > 0.0);
    }
}
