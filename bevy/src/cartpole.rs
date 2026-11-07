//! CartPole with Avian doing the physics: a cart on a slider joint along a fixed track, a
//! pole on a hinge on the cart, gravity, and a push left or right each step. The rules are
//! gym's (12 degrees, 2.4 m, +0.01 a step, max_steps); the dynamics are Avian's solver
//! rather than gym's Euler step, so the numbers are close to gym's but not the same.
//!
//! Observations: cart position, cart velocity, pole angle, pole angular velocity, scaled
//! to about [-1, 1]. Actions: one head, 0 push left, 1 push right.

use avian2d::prelude::*;
use bevy::ecs::entity::Entity;
use bevy::ecs::world::World;
use bevy::color::Color;
use bevy::math::Vec2;
use pufferl::env::{Config, Log};

use crate::view::Mark;
use crate::{placed, teleport, BevyEnv, Rng};

const FORCE: f32 = 10.0;
const X_LIMIT: f32 = 2.4;
const THETA_LIMIT: f32 = 12.0 * std::f32::consts::PI / 180.0;
const REWARD: f32 = 0.01;
const CART: Vec2 = Vec2::new(0.5, 0.3);
/// The pole is 1 m from hinge to tip, so its center is half that up
const POLE: Vec2 = Vec2::new(0.1, 1.0);

pub struct CartPole {
    max_steps: usize,
}

pub struct Bodies {
    cart: Entity,
    pole: Entity,
}

impl CartPole {
    pub fn new(config: &Config) -> Self {
        Self { max_steps: config.size("max_steps") }
    }

    fn state(world: &World, arena: &Bodies, origin: Vec2) -> [f32; 4] {
        let x = world.get::<Position>(arena.cart).unwrap().x - origin.x;
        let x_dot = world.get::<LinearVelocity>(arena.cart).unwrap().x;
        // Positive leans right, as gym has it: Avian's angles go counterclockwise
        let theta = -world.get::<Rotation>(arena.pole).unwrap().as_radians();
        let theta_dot = -world.get::<AngularVelocity>(arena.pole).unwrap().0;
        [x, x_dot, theta, theta_dot]
    }
}

impl BevyEnv for CartPole {
    type Arena = Bodies;

    fn obs_size(&self) -> usize {
        4
    }

    fn act_sizes(&self) -> Vec<usize> {
        vec![2]
    }

    /// Track ends 2.4 m out and a pole 1 m tall: 6 m apart, arenas never meet
    fn spacing(&self) -> f32 {
        6.0
    }

    /// The track, and its ends
    fn marks(&self, _world: &World, _arena: &Bodies, origin: Vec2, marks: &mut Vec<Mark>) {
        let color = Color::srgb(0.55, 0.57, 0.62);
        let y = origin.y - CART.y / 2.0;
        marks.push(Mark::Line { from: Vec2::new(origin.x - X_LIMIT, y), to: Vec2::new(origin.x + X_LIMIT, y), color });
        for end in [-X_LIMIT, X_LIMIT] {
            marks.push(Mark::Line { from: Vec2::new(origin.x + end, y - 0.15), to: Vec2::new(origin.x + end, y + 0.15), color });
        }
    }

    fn spawn(&self, world: &mut World, origin: Vec2) -> Bodies {
        let track = world.spawn((RigidBody::Static, placed(origin, 0.0))).id();
        // Mass from area: cart 1 kg, pole 0.1 kg, as gym's are
        let cart = world
            .spawn((
                RigidBody::Dynamic,
                Collider::rectangle(CART.x, CART.y),
                ColliderDensity(1.0 / (CART.x * CART.y)),
                placed(origin, 0.0),
                ConstantForce::new(0.0, 0.0),
            ))
            .id();
        let pole = world
            .spawn((
                RigidBody::Dynamic,
                Collider::rectangle(POLE.x, POLE.y),
                ColliderDensity(0.1 / (POLE.x * POLE.y)),
                placed(origin + Vec2::Y * POLE.y / 2.0, 0.0),
            ))
            .id();
        world.spawn(PrismaticJoint::new(track, cart).with_slider_axis(Vec2::X));
        world.spawn((RevoluteJoint::new(cart, pole).with_local_anchor2(Vec2::new(0.0, -POLE.y / 2.0)), JointCollisionDisabled));
        Bodies { cart, pole }
    }

    fn reset(&self, world: &mut World, arena: &mut Bodies, origin: Vec2, rng: &mut Rng) {
        let mut jitter = || rng.range(-0.05, 0.05);
        let (x, x_dot, theta, theta_dot) = (jitter(), jitter(), jitter(), jitter());
        let cart = origin + Vec2::X * x;
        teleport(world, arena.cart, cart, 0.0, Vec2::X * x_dot, 0.0);
        // The pole's center, half its length up it from the hinge
        let lean = Vec2::new(theta.sin(), theta.cos()) * POLE.y / 2.0;
        let swing = Vec2::new(theta.cos(), -theta.sin()) * theta_dot * POLE.y / 2.0;
        teleport(world, arena.pole, cart + lean, -theta, Vec2::X * x_dot + swing, -theta_dot);
        world.get_mut::<ConstantForce>(arena.cart).unwrap().0 = Vec2::ZERO;
    }

    fn act(&self, world: &mut World, arena: &mut Bodies, actions: &[usize]) {
        let push = if actions[0] == 1 { FORCE } else { -FORCE };
        world.get_mut::<ConstantForce>(arena.cart).unwrap().0 = Vec2::X * push;
    }

    fn observe(&self, world: &World, arena: &Bodies, origin: Vec2, obs: &mut [f32]) {
        let [x, x_dot, theta, theta_dot] = Self::state(world, arena, origin);
        obs.copy_from_slice(&[x / X_LIMIT, x_dot / 2.0, theta / THETA_LIMIT, theta_dot / 2.0]);
    }

    fn outcome(&self, world: &World, arena: &mut Bodies, origin: Vec2, tick: usize) -> (f32, bool) {
        let [x, _, theta, _] = Self::state(world, arena, origin);
        let fell = x.abs() > X_LIMIT || theta.abs() > THETA_LIMIT;
        (REWARD, fell || tick >= self.max_steps)
    }

    fn log_episode(&self, log: &mut Log, _arena: &Bodies, _episode_return: f64, length: usize) {
        let length = length as f64;
        log.episode(&[
            ("perf", length / self.max_steps as f64),
            ("score", length),
            ("episode_return", length * REWARD as f64),
            ("episode_length", length),
        ]);
    }
}
