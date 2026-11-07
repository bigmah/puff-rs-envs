//! Pusher: a top-down walled room, a round agent, a box, and a goal on the floor. The agent
//! pushes itself about; the box moves only when something hits it, the agent or a wall.
//! Getting the box's center onto the goal is +1 and ends the episode. On the way, a step
//! pays for bringing the box nearer the goal, and a little for bringing the agent nearer
//! the box. max_steps without it ends the episode with nothing more.
//!
//! Observations: the agent's place and velocity, the box's, and the goal's place, relative
//! to the room's center and scaled to about [-1, 1].
//! Actions: one head, 0 nothing, 1 up, 2 down, 3 left, 4 right.

use avian2d::prelude::*;
use bevy::app::App;
use bevy::ecs::entity::Entity;
use bevy::ecs::world::World;
use bevy::color::Color;
use bevy::math::Vec2;
use pufferl::env::{Config, Log};

use crate::view::Mark;
use crate::{placed, teleport, BevyEnv, Rng};

/// Half the room's inside width
const HALF: f32 = 4.0;
const WALL: f32 = 0.5;
const AGENT_RADIUS: f32 = 0.3;
const BOX_SIDE: f32 = 0.8;
const GOAL_RADIUS: f32 = 0.8;
const FORCE: f32 = 20.0;
const BOX_SHAPING: f32 = 1.0;
const REACH_SHAPING: f32 = 0.2;

pub struct Pusher {
    max_steps: usize,
}

pub struct Room {
    agent: Entity,
    block: Entity,
    goal: Vec2,
    box_to_goal: f32,
    agent_to_box: f32,
    solved: bool,
}

impl Pusher {
    pub fn new(config: &Config) -> Self {
        Self { max_steps: config.size("max_steps") }
    }
}

fn place(world: &World, entity: Entity) -> Vec2 {
    world.get::<Position>(entity).unwrap().0
}

fn velocity(world: &World, entity: Entity) -> Vec2 {
    world.get::<LinearVelocity>(entity).unwrap().0
}

impl BevyEnv for Pusher {
    type Arena = Room;

    fn obs_size(&self) -> usize {
        10
    }

    fn act_sizes(&self) -> Vec<usize> {
        vec![5]
    }

    /// The room and its walls are 9 m across
    fn spacing(&self) -> f32 {
        10.0
    }

    /// The goal
    fn marks(&self, _world: &World, room: &Room, _origin: Vec2, marks: &mut Vec<Mark>) {
        marks.push(Mark::Circle { at: room.goal, radius: GOAL_RADIUS, color: Color::srgb(0.2, 0.85, 0.4) });
    }

    fn build(&self, app: &mut App) {
        // Seen from above: nothing falls
        app.insert_resource(Gravity(Vec2::ZERO));
    }

    fn spawn(&self, world: &mut World, origin: Vec2) -> Room {
        for (at, size) in [
            (Vec2::new(0.0, HALF + WALL / 2.0), Vec2::new(2.0 * HALF + 2.0 * WALL, WALL)),
            (Vec2::new(0.0, -HALF - WALL / 2.0), Vec2::new(2.0 * HALF + 2.0 * WALL, WALL)),
            (Vec2::new(HALF + WALL / 2.0, 0.0), Vec2::new(WALL, 2.0 * HALF)),
            (Vec2::new(-HALF - WALL / 2.0, 0.0), Vec2::new(WALL, 2.0 * HALF)),
        ] {
            world.spawn((RigidBody::Static, Collider::rectangle(size.x, size.y), placed(origin + at, 0.0)));
        }
        let block = world
            .spawn((
                RigidBody::Dynamic,
                Collider::rectangle(BOX_SIDE, BOX_SIDE),
                ColliderDensity(2.0),
                Friction::new(0.3),
                LinearDamping(3.0),
                AngularDamping(3.0),
                placed(origin + Vec2::X, 0.0),
            ))
            .id();
        let agent = world
            .spawn((
                RigidBody::Dynamic,
                Collider::circle(AGENT_RADIUS),
                ColliderDensity(4.0),
                LockedAxes::ROTATION_LOCKED,
                LinearDamping(4.0),
                placed(origin - Vec2::X, 0.0),
                ConstantForce::new(0.0, 0.0),
            ))
            .id();
        Room { agent, block, goal: origin, box_to_goal: 0.0, agent_to_box: 0.0, solved: false }
    }

    /// The goal anywhere, the box 1 to 2.5 m from it, the agent 0.8 to 2 m from the box
    fn reset(&self, world: &mut World, room: &mut Room, origin: Vec2, rng: &mut Rng) {
        let inside = HALF - BOX_SIDE;
        let mut point = || Vec2::new(rng.range(-inside, inside), rng.range(-inside, inside));
        let goal = point();
        let block = loop {
            let p = point();
            if (1.0..2.5).contains(&p.distance(goal)) {
                break p;
            }
        };
        let agent = loop {
            let p = point();
            if (0.8..2.0).contains(&p.distance(block)) {
                break p;
            }
        };
        let turn = rng.range(0.0, std::f32::consts::FRAC_PI_2);
        teleport(world, room.block, origin + block, turn, Vec2::ZERO, 0.0);
        teleport(world, room.agent, origin + agent, 0.0, Vec2::ZERO, 0.0);
        world.get_mut::<ConstantForce>(room.agent).unwrap().0 = Vec2::ZERO;
        room.goal = origin + goal;
        (room.box_to_goal, room.agent_to_box, room.solved) = (block.distance(goal), agent.distance(block), false);
    }

    fn act(&self, world: &mut World, room: &mut Room, actions: &[usize]) {
        let push = match actions[0] {
            1 => Vec2::Y,
            2 => Vec2::NEG_Y,
            3 => Vec2::NEG_X,
            4 => Vec2::X,
            _ => Vec2::ZERO,
        };
        world.get_mut::<ConstantForce>(room.agent).unwrap().0 = push * FORCE;
    }

    fn observe(&self, world: &World, room: &Room, origin: Vec2, obs: &mut [f32]) {
        let (agent, block) = (place(world, room.agent) - origin, place(world, room.block) - origin);
        let (agent_v, block_v) = (velocity(world, room.agent), velocity(world, room.block));
        let goal = room.goal - origin;
        obs.copy_from_slice(&[
            agent.x / HALF, agent.y / HALF, agent_v.x / 4.0, agent_v.y / 4.0,
            block.x / HALF, block.y / HALF, block_v.x / 4.0, block_v.y / 4.0,
            goal.x / HALF, goal.y / HALF,
        ]);
    }

    fn outcome(&self, world: &World, room: &mut Room, _origin: Vec2, tick: usize) -> (f32, bool) {
        let (agent, block) = (place(world, room.agent), place(world, room.block));
        let (box_to_goal, agent_to_box) = (block.distance(room.goal), agent.distance(block));
        let mut reward = BOX_SHAPING * (room.box_to_goal - box_to_goal) + REACH_SHAPING * (room.agent_to_box - agent_to_box);
        (room.box_to_goal, room.agent_to_box) = (box_to_goal, agent_to_box);
        room.solved = box_to_goal < GOAL_RADIUS;
        if room.solved {
            reward += 1.0;
        }
        (reward, room.solved || tick >= self.max_steps)
    }

    fn log_episode(&self, log: &mut Log, room: &Room, episode_return: f64, length: usize) {
        let solved = if room.solved { 1.0 } else { 0.0 };
        log.episode(&[
            ("perf", solved),
            ("score", solved),
            ("episode_return", episode_return),
            ("episode_length", length as f64),
        ]);
    }
}
