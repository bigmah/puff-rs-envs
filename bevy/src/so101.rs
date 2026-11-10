//! SO-101: the SO-ARM101 arm seen from the side, picking a cube up off a table and carrying
//! it to goals in the air, one after another.
//!
//! The arm is the real one's planar joints at its real size and weight (from the SO-101
//! URDF): shoulder lift 11.7 cm above the table, an 11.6 cm upper arm, a 13.5 cm forearm,
//! and a hand 16 cm from wrist to fingertips. Each joint is an STS3215 servo: a position
//! motor with a stiff spring, damped, and no more than 3 N·m, within the real joint's
//! limits. The base's pan and the wrist's roll turn out of the plane, so they are not
//! here. The gripper is the hand's flat face and a jaw that slides against it, a parallel
//! stand-in for the real arm's swinging jaw, pushed shut or open by a force as a
//! current-limited servo pushes: the cube is held by squeeze and friction alone, with
//! nothing gluing it on.
//!
//! Each episode the cube starts somewhere on the table in reach, and a goal hangs in the
//! air. Carrying the cube, held, to within 2 cm of the goal is +5, and a new goal appears;
//! max_steps (50 a second) ends the episode. Knocking the cube off the table ends it too,
//! for -2. On the way a step pays for bringing the fingers nearer the cube, for having it
//! held, and for bringing the cube nearer the goal, all as a change of potential, so
//! grabbing and letting go over and over gains nothing.
//!
//! Observations: joint angles, speeds and how far each servo's target is from its angle,
//! the gripper's opening and command, where the fingers are and which way the hand points,
//! the cube's place, turn and velocity, the goal, the cube from the fingers and the goal
//! from the cube, and whether the cube is held. Places are from the shoulder's foot on the
//! table, scaled to about [-1, 1].
//!
//! Actions: four heads. Shoulder, elbow and wrist each move their servo's target 0 fast
//! down, 1 slowly down, 2 not, 3 slowly up or 4 fast up; the gripper 0 opens, 1 closes.

use std::f32::consts::{FRAC_PI_2, PI};

use puffer_bevy::avian2d::dynamics::solver::SolverConfig;
use puffer_bevy::avian2d::prelude::*;
use puffer_bevy::bevy::app::App;
use puffer_bevy::bevy::color::Color;
use puffer_bevy::bevy::ecs::entity::Entity;
use puffer_bevy::bevy::ecs::world::World;
use puffer_bevy::bevy::math::{Rot2, Vec2};
use pufferl::env::{Config, Log};

use puffer_bevy::view::Mark;
use puffer_bevy::{placed, teleport, BevyEnv, Rng};

/// Shoulder lift's axis above the table
const SHOULDER_H: f32 = 0.1166;
const BASE_W: f32 = 0.05;
const UPPER: f32 = 0.116;
const FORE: f32 = 0.135;
const LINK_W: f32 = 0.026;
/// Wrist axis to fingertips, and the hand's thickness behind its gripping face
const HAND: f32 = 0.16;
const HAND_W: f32 = 0.03;
/// The moving jaw, and the length of hand it grips against, at the hand's end
const JAW: f32 = 0.055;
const JAW_W: f32 = 0.01;
const OPEN_MAX: f32 = 0.06;
const CUBE: f32 = 0.032;
/// The grip point, wrist to between the jaws a cube's half and a little from the tips
const GRIP: f32 = HAND - CUBE / 2.0 - 0.005;

/// Masses from the URDF: wrist and gripper together are the hand. The jaw is the real
/// one's 12 g and its slide's
const UPPER_KG: f32 = 0.103;
const FORE_KG: f32 = 0.104;
const HAND_KG: f32 = 0.166;
const JAW_KG: f32 = 0.04;
const CUBE_KG: f32 = 0.03;

/// The URDF's limits, around a zero of the upper arm straight up, the forearm square to
/// it pointing forward, and the hand in line with the forearm
const LIMITS: [f32; 3] = [1.745, 1.69, 1.658];
/// STS3215 at 12 V stalls at about 30 kg·cm
const MAX_TORQUE: f32 = 3.0;
/// The gripper pushes its jaw shut or open with this, less this much a m/s of the jaw's
/// speed: a current-limited servo, which squeezes what it holds with GRIP_FORCE, and
/// opens or shuts at up to 0.4 m/s
const GRIP_FORCE: f32 = 0.6;
const GRIP_DAMPING: f32 = 1.5;
/// Avian's acceleration-based joint motor: stiff enough that the arm sags about 2 degrees
/// holding itself out, as a real STS3215 does
const SERVO_K: f32 = 120000.0;
const SERVO_D: f32 = 0.9;
/// Target speeds, rad/s, fast and slow
const FAST: f32 = 2.5;
const SLOW: f32 = 0.6;
/// A servo's target stays this near its angle, so pushing on the table doesn't wind it up
const LEAD: f32 = 0.25;

/// Where the cube starts and goals are, forward of the shoulder
const REACH: (f32, f32) = (0.12, 0.30);
const GOAL_Y: (f32, f32) = (0.05, 0.20);
const GOAL_RADIUS: f32 = 0.02;
const TABLE: (f32, f32) = (-0.15, 0.48);

/// Shaping a meter, and for holding
const REACH_SHAPING: f32 = 10.0;
const HELD_SHAPING: f32 = 1.0;
const CARRY_SHAPING: f32 = 10.0;
const GOAL: f32 = 5.0;
const DROPPED: f32 = -2.0;

/// Contacts stiffer than Avian's default (1.5 and 4), or the jaw sinks into the cube and
/// it slips out: a scripted grasp gets a fifth of the goals
const CONTACT_HZ: f32 = 4.0;
const OVERLAP_SPEED: f32 = 40.0;

/// Collision layers: the arm and hand touch the table and the cube, not each other
const L_TABLE: u32 = 1;
const L_ARM: u32 = 2;
const L_BASE: u32 = 4;
const L_CUBE: u32 = 8;

pub struct So101 {
    max_steps: usize,
}

pub struct Cell {
    upper: Entity,
    fore: Entity,
    hand: Entity,
    jaw: Entity,
    cube: Entity,
    /// shoulder, elbow, wrist
    joints: [Entity; 3],
    targets: [f32; 3],
    close: bool,
    goal: Vec2,
    potential: f32,
    goals: u32,
    ever_held: bool,
    dropped: bool,
    /// For goals after the first, which come mid-episode
    rng: Rng,
}

impl So101 {
    pub fn new(config: &Config) -> Self {
        Self { max_steps: config.size("max_steps") }
    }
}

fn dir(angle: f32) -> Vec2 {
    Vec2::new(angle.cos(), angle.sin())
}

fn place(world: &World, entity: Entity) -> Vec2 {
    world.get::<Position>(entity).unwrap().0
}

fn angle(world: &World, entity: Entity) -> f32 {
    world.get::<Rotation>(entity).unwrap().as_radians()
}

fn spin(world: &World, entity: Entity) -> f32 {
    world.get::<AngularVelocity>(entity).unwrap().0
}

fn wrap(a: f32) -> f32 {
    (a + PI).rem_euclid(2.0 * PI) - PI
}

/// A dynamic box of the arm: mass kg over its area
fn part(size: Vec2, kg: f32, at: Vec2, turn: f32, layers: CollisionLayers) -> impl puffer_bevy::bevy::ecs::bundle::Bundle {
    (
        RigidBody::Dynamic,
        Collider::rectangle(size.x, size.y),
        ColliderDensity(kg / (size.x * size.y)),
        layers,
        Friction::new(1.0),
        Restitution::new(0.0),
        SleepingDisabled,
        placed(at, turn),
    )
}

/// The arm's world angles for joint angles q: upper arm, forearm, hand
fn link_angles(q: [f32; 3]) -> [f32; 3] {
    let a1 = FRAC_PI_2 + q[0];
    let a2 = a1 - FRAC_PI_2 + q[1];
    [a1, a2, a2 + q[2]]
}

/// What the arm knows of itself
struct Pose {
    q: [f32; 3],
    qd: [f32; 3],
    /// The hand's angle, the wrist, the gripper's opening
    hand_angle: f32,
    wrist: Vec2,
    opening: f32,
}

impl Pose {
    fn read(world: &World, cell: &Cell, origin: Vec2) -> Self {
        let (a1, a2, a3) = (angle(world, cell.upper), angle(world, cell.fore), angle(world, cell.hand));
        let q = [wrap(a1 - FRAC_PI_2), wrap(a2 - a1 + FRAC_PI_2), wrap(a3 - a2)];
        let (w1, w2, w3) = (spin(world, cell.upper), spin(world, cell.fore), spin(world, cell.hand));
        let hand = place(world, cell.hand) - origin;
        let rot = Rot2::radians(a3);
        let wrist = hand - rot * Vec2::new(HAND / 2.0, -HAND_W / 2.0);
        let jaw = rot.inverse() * (place(world, cell.jaw) - origin - wrist);
        Pose { q, qd: [w1, w2 - w1, w3 - w2], hand_angle: a3, wrist, opening: jaw.y - JAW_W / 2.0 }
    }

    /// Where between the jaws a cube is gripped, near the fingertips, so a cube on the
    /// table can have its center there
    fn fingers(&self) -> Vec2 {
        self.wrist + Rot2::radians(self.hand_angle) * Vec2::new(GRIP, self.opening / 2.0)
    }

    /// The cube in the hand's frame, from the wrist
    fn in_hand(&self, cube: Vec2) -> Vec2 {
        Rot2::radians(self.hand_angle).inverse() * (cube - self.wrist)
    }
}

/// The cube is held when it's between the jaws, they're closing, and they're on it: open
/// as wide as the cube is across them, turned as it is in the hand
fn held(pose: &Pose, cube: Vec2, cube_angle: f32, close: bool) -> bool {
    let c = pose.in_hand(cube);
    let turn = cube_angle - pose.hand_angle;
    let across = CUBE * (turn.cos().abs() + turn.sin().abs());
    close
        && (HAND - JAW - CUBE * 0.2..=HAND + CUBE * 0.2).contains(&c.x)
        && (0.0..=pose.opening).contains(&c.y)
        && (across - 0.006..=across + 0.003).contains(&pose.opening)
}

impl So101 {
    fn potential(&self, world: &World, cell: &Cell, origin: Vec2) -> (f32, bool) {
        let pose = Pose::read(world, cell, origin);
        let cube = place(world, cell.cube) - origin;
        let holding = held(&pose, cube, angle(world, cell.cube), cell.close);
        let phi = -REACH_SHAPING * pose.fingers().distance(cube) + if holding { HELD_SHAPING } else { 0.0 }
            - CARRY_SHAPING * cube.distance(cell.goal);
        (phi, holding)
    }

    fn new_goal(rng: &mut Rng, cube: Vec2) -> Vec2 {
        loop {
            let goal = Vec2::new(rng.range(REACH.0, REACH.1), rng.range(GOAL_Y.0, GOAL_Y.1));
            if goal.distance(cube) > 0.06 {
                return goal;
            }
        }
    }
}

impl BevyEnv for So101 {
    type Arena = Cell;

    fn obs_size(&self) -> usize {
        28
    }

    fn act_sizes(&self) -> Vec<usize> {
        vec![5, 5, 5, 2]
    }

    /// The table is 63 cm long and the arm reaches 41 cm up
    fn spacing(&self) -> f32 {
        0.8
    }

    fn build(&self, app: &mut App) {
        // The arm is centimeters across, not meters: Avian's tolerances scale with this
        app.insert_resource(PhysicsLengthUnit(0.05));
        app.insert_resource(SubstepCount(8));
        app.insert_resource(SolverConfig { contact_frequency_factor: CONTACT_HZ, max_overlap_solve_speed: OVERLAP_SPEED, ..Default::default() });
    }

    /// The goal: green while the cube is held, yellow while not
    fn marks(&self, world: &World, cell: &Cell, origin: Vec2, marks: &mut Vec<Mark>) {
        let held_now = self.potential(world, cell, origin).1;
        let color = if held_now { Color::srgb(0.2, 0.85, 0.4) } else { Color::srgb(0.9, 0.8, 0.2) };
        marks.push(Mark::Circle { at: origin + cell.goal, radius: GOAL_RADIUS, color });
    }

    fn spawn(&self, world: &mut World, origin: Vec2) -> Cell {
        let layers = |member: u32, filter: u32| CollisionLayers::from_bits(member, filter);
        let table_len = TABLE.1 - TABLE.0;
        world.spawn((
            RigidBody::Static,
            Collider::rectangle(table_len, 0.04),
            layers(L_TABLE, L_ARM | L_CUBE),
            Friction::new(0.6),
            placed(origin + Vec2::new((TABLE.0 + TABLE.1) / 2.0, -0.02), 0.0),
        ));
        let base = world
            .spawn((RigidBody::Static, Collider::rectangle(BASE_W, SHOULDER_H), layers(L_BASE, L_CUBE), placed(origin + Vec2::Y * SHOULDER_H / 2.0, 0.0)))
            .id();

        let arm = layers(L_ARM, L_TABLE | L_CUBE);
        let upper = world.spawn(part(Vec2::new(UPPER, LINK_W), UPPER_KG, origin, 0.0, arm)).id();
        let fore = world.spawn(part(Vec2::new(FORE, LINK_W), FORE_KG, origin, 0.0, arm)).id();
        let hand = world.spawn((part(Vec2::new(HAND, HAND_W), HAND_KG, origin, 0.0, arm), ConstantForce::new(0.0, 0.0))).id();
        let jaw = world.spawn((part(Vec2::new(JAW, JAW_W), JAW_KG, origin, 0.0, arm), ConstantForce::new(0.0, 0.0))).id();
        let cube = world
            .spawn((
                RigidBody::Dynamic,
                Collider::rectangle(CUBE, CUBE),
                ColliderDensity(CUBE_KG / (CUBE * CUBE)),
                layers(L_CUBE, L_TABLE | L_ARM | L_BASE),
                Friction::new(1.0),
                Restitution::new(0.0),
                SleepingDisabled,
                placed(origin + Vec2::new(0.2, CUBE / 2.0), 0.0),
            ))
            .id();

        let motor = || AngularMotor::new(MotorModel::AccelerationBased { stiffness: SERVO_K, damping: SERVO_D }).with_max_torque(MAX_TORQUE);
        let hinge = |body1: Entity, anchor1: Vec2, basis1: f32, body2: Entity, anchor2: Vec2, limit: f32| {
            (
                RevoluteJoint::new(body1, body2)
                    .with_local_anchor1(anchor1)
                    .with_local_anchor2(anchor2)
                    .with_local_basis1(Rot2::radians(basis1))
                    .with_local_basis2(Rot2::IDENTITY)
                    .with_angle_limits(-limit, limit)
                    .with_motor(motor()),
                JointCollisionDisabled,
            )
        };
        let shoulder = world.spawn(hinge(base, Vec2::Y * SHOULDER_H / 2.0, FRAC_PI_2, upper, Vec2::NEG_X * UPPER / 2.0, LIMITS[0])).id();
        let elbow = world.spawn(hinge(upper, Vec2::X * UPPER / 2.0, -FRAC_PI_2, fore, Vec2::NEG_X * FORE / 2.0, LIMITS[1])).id();
        let wrist = world.spawn(hinge(fore, Vec2::X * FORE / 2.0, 0.0, hand, Vec2::new(-HAND / 2.0, HAND_W / 2.0), LIMITS[2])).id();
        // The jaw slides square to the hand's face, its inner side opening from it
        world.spawn((
            PrismaticJoint::new(hand, jaw)
                .with_slider_axis(Vec2::Y)
                .with_local_anchor1(Vec2::new(HAND / 2.0 - JAW / 2.0, HAND_W / 2.0 + JAW_W / 2.0))
                .with_local_anchor2(Vec2::ZERO)
                .with_local_basis1(Rot2::IDENTITY)
                .with_local_basis2(Rot2::IDENTITY)
                .with_limits(0.0, OPEN_MAX),
            JointCollisionDisabled,
        ));

        Cell {
            upper,
            fore,
            hand,
            jaw,
            cube,
            joints: [shoulder, elbow, wrist],
            targets: [0.0; 3],
            close: false,
            goal: Vec2::new(0.2, 0.1),
            potential: 0.0,
            goals: 0,
            ever_held: false,
            dropped: false,
            rng: Rng::new(0),
        }
    }

    /// The arm near its rest pose, the gripper some way open, the cube anywhere on the
    /// table in reach, and a goal in the air
    fn reset(&self, world: &mut World, cell: &mut Cell, origin: Vec2, rng: &mut Rng) {
        let q = [rng.range(-0.2, 0.2), rng.range(0.0, 0.4), rng.range(-0.4, 0.2)];
        let opening = rng.range(0.0, OPEN_MAX);
        let [a1, a2, a3] = link_angles(q);
        let shoulder = origin + Vec2::Y * SHOULDER_H;
        let elbow = shoulder + dir(a1) * UPPER;
        let wrist = elbow + dir(a2) * FORE;
        let rot = Rot2::radians(a3);
        teleport(world, cell.upper, shoulder + dir(a1) * UPPER / 2.0, a1, Vec2::ZERO, 0.0);
        teleport(world, cell.fore, elbow + dir(a2) * FORE / 2.0, a2, Vec2::ZERO, 0.0);
        teleport(world, cell.hand, wrist + rot * Vec2::new(HAND / 2.0, -HAND_W / 2.0), a3, Vec2::ZERO, 0.0);
        teleport(world, cell.jaw, wrist + rot * Vec2::new(HAND - JAW / 2.0, opening + JAW_W / 2.0), a3, Vec2::ZERO, 0.0);

        let cube = Vec2::new(rng.range(REACH.0, REACH.1), CUBE / 2.0);
        let turn = rng.range(-0.3, 0.3);
        teleport(world, cell.cube, origin + cube, turn, Vec2::ZERO, 0.0);

        for (joint, &target) in cell.joints.iter().zip(&q) {
            world.get_mut::<RevoluteJoint>(*joint).unwrap().motor.target_position = target;
        }
        for body in [cell.hand, cell.jaw] {
            world.get_mut::<ConstantForce>(body).unwrap().0 = Vec2::ZERO;
        }
        cell.targets = q;
        cell.close = false;
        cell.goal = Self::new_goal(rng, cube);
        cell.rng = Rng::new(rng.next_u64());
        (cell.goals, cell.ever_held, cell.dropped) = (0, false, false);
        cell.potential = self.potential(world, cell, origin).0;
    }

    fn act(&self, world: &mut World, cell: &mut Cell, actions: &[usize]) {
        let dt = self.dt();
        let q = {
            let (a1, a2, a3) = (angle(world, cell.upper), angle(world, cell.fore), angle(world, cell.hand));
            [wrap(a1 - FRAC_PI_2), wrap(a2 - a1 + FRAC_PI_2), wrap(a3 - a2)]
        };
        for i in 0..3 {
            let speed = [-FAST, -SLOW, 0.0, SLOW, FAST][actions[i]];
            let target = (cell.targets[i] + speed * dt).clamp(q[i] - LEAD, q[i] + LEAD).clamp(-LIMITS[i], LIMITS[i]);
            cell.targets[i] = target;
            world.get_mut::<RevoluteJoint>(cell.joints[i]).unwrap().motor.target_position = target;
        }
        cell.close = actions[3] == 1;
        // Along the jaw's slide: open is the hand's +y
        let slide = Rot2::radians(angle(world, cell.hand)) * Vec2::Y;
        let speed = (world.get::<LinearVelocity>(cell.jaw).unwrap().0 - world.get::<LinearVelocity>(cell.hand).unwrap().0).dot(slide);
        let push = (if cell.close { -GRIP_FORCE } else { GRIP_FORCE }) - GRIP_DAMPING * speed;
        world.get_mut::<ConstantForce>(cell.jaw).unwrap().0 = slide * push;
        world.get_mut::<ConstantForce>(cell.hand).unwrap().0 = -slide * push;
    }

    fn observe(&self, world: &World, cell: &Cell, origin: Vec2, obs: &mut [f32]) {
        let pose = Pose::read(world, cell, origin);
        let cube = place(world, cell.cube) - origin;
        let cube_turn = 4.0 * angle(world, cell.cube);
        let cube_v = world.get::<LinearVelocity>(cell.cube).unwrap().0;
        let fingers = pose.fingers();
        let holding = held(&pose, cube, angle(world, cell.cube), cell.close);
        let s = 1.0 / REACH.1;
        let (to_cube, to_goal) = ((cube - fingers) * 10.0, (cell.goal - cube) * 10.0);
        obs.copy_from_slice(&[
            pose.q[0] / LIMITS[0], pose.q[1] / LIMITS[1], pose.q[2] / LIMITS[2],
            pose.qd[0] / 5.0, pose.qd[1] / 5.0, pose.qd[2] / 5.0,
            (cell.targets[0] - pose.q[0]) / LEAD, (cell.targets[1] - pose.q[1]) / LEAD, (cell.targets[2] - pose.q[2]) / LEAD,
            pose.opening / OPEN_MAX, if cell.close { 1.0 } else { 0.0 },
            fingers.x * s, fingers.y * s, pose.hand_angle.cos(), pose.hand_angle.sin(),
            cube.x * s, cube.y * s, cube_turn.cos(), cube_turn.sin(), cube_v.x, cube_v.y,
            cell.goal.x * s, cell.goal.y * s,
            to_cube.x.clamp(-3.0, 3.0), to_cube.y.clamp(-3.0, 3.0), to_goal.x.clamp(-3.0, 3.0), to_goal.y.clamp(-3.0, 3.0),
            if holding { 1.0 } else { 0.0 },
        ]);
    }

    fn outcome(&self, world: &World, cell: &mut Cell, origin: Vec2, tick: usize) -> (f32, bool) {
        let (phi, holding) = self.potential(world, cell, origin);
        let mut reward = phi - cell.potential;
        cell.potential = phi;
        cell.ever_held |= holding;
        let cube = place(world, cell.cube) - origin;
        if holding && cube.distance(cell.goal) < GOAL_RADIUS {
            reward += GOAL;
            cell.goals += 1;
            // The next goal, with a fresh potential so its distance isn't a penalty
            cell.goal = Self::new_goal(&mut cell.rng, cube);
            cell.potential = self.potential(world, cell, origin).0;
        }
        if cube.y < -0.01 || !(TABLE.0..TABLE.1).contains(&cube.x) {
            cell.dropped = true;
            return (reward + DROPPED, true);
        }
        (reward, tick >= self.max_steps)
    }

    fn log_episode(&self, log: &mut Log, cell: &Cell, episode_return: f64, length: usize) {
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        log.episode(&[
            ("perf", flag(cell.goals > 0)),
            ("score", cell.goals as f64),
            ("grasped", flag(cell.ever_held)),
            ("dropped", flag(cell.dropped)),
            ("episode_return", episode_return),
            ("episode_length", length as f64),
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use puffer_bevy::bench::{config, Buffers};
    use puffer_bevy::Arenas;
    use pufferl::env::Env;

    /// Step every arena n times with the actions policy gives its observation; the
    /// observations after, and the reward over them all
    fn run(env: &mut Arenas<So101>, buffers: &mut Buffers, n: usize, policy: impl Fn(&[f32]) -> [usize; 4]) -> f32 {
        let mut reward = 0.0;
        for _ in 0..n {
            let obs_size = buffers.obs_size;
            for (actions, obs) in buffers.actions.chunks_mut(4).zip(buffers.obs.chunks(obs_size)) {
                for (a, b) in actions.iter_mut().zip(policy(obs)) {
                    *a = b as f32;
                }
            }
            env.step(buffers.agents());
            reward += buffers.rewards.iter().sum::<f32>();
        }
        reward
    }

    fn joints(obs: &[f32]) -> [f32; 3] {
        [obs[0] * LIMITS[0], obs[1] * LIMITS[1], obs[2] * LIMITS[2]]
    }

    /// The joint angles that put the fingers at f with the hand pointing down, elbow up,
    /// for a gripper opening: straight down, or tilted forward as far as the wrist needs
    fn ik(f: Vec2, opening: f32) -> [f32; 3] {
        let mut a3 = -FRAC_PI_2;
        let mut q = [0.0; 3];
        for _ in 0..4 {
            let wrist = f - Rot2::radians(a3) * Vec2::new(GRIP, opening / 2.0);
            let to = wrist - Vec2::Y * SHOULDER_H;
            let d = to.length().clamp(0.03, UPPER + FORE - 1e-3);
            let inner = ((UPPER * UPPER + d * d - FORE * FORE) / (2.0 * UPPER * d)).clamp(-1.0, 1.0).acos();
            let a1 = to.y.atan2(to.x) + inner;
            let elbow = Vec2::Y * SHOULDER_H + dir(a1) * UPPER;
            let a2 = (wrist - elbow).y.atan2((wrist - elbow).x);
            q = [wrap(a1 - FRAC_PI_2), wrap(a2 - a1 + FRAC_PI_2), wrap(a3 - a2)];
            a3 = (a2 - (LIMITS[2] - 0.1)).max(-FRAC_PI_2);
        }
        q
    }

    /// Move each servo's target toward q: fast when far, slowly when near
    fn toward(obs: &[f32], q: [f32; 3]) -> [usize; 3] {
        let now = joints(obs);
        std::array::from_fn(|i| {
            let e = q[i] - (now[i] + obs[6 + i] * LEAD);
            if e > 0.08 { 4 } else if e > 0.01 { 3 } else if e < -0.08 { 0 } else if e < -0.01 { 1 } else { 2 }
        })
    }

    /// Over the cube, down to it, close, carry it to the goal
    fn scripted(obs: &[f32]) -> [usize; 4] {
        let s = REACH.1;
        let (fingers, cube, goal) = (Vec2::new(obs[11], obs[12]) * s, Vec2::new(obs[15], obs[16]) * s, Vec2::new(obs[21], obs[22]) * s);
        let opening = obs[9] * OPEN_MAX;
        let closing = obs[10] > 0.5 && opening > CUBE * 0.8 && fingers.distance(cube) < 0.03;
        let (target, close) = if obs[27] > 0.5 {
            (goal, true)
        } else if closing {
            (cube, true)
        } else if (cube.x - fingers.x).abs() > 0.01 {
            (cube + Vec2::Y * 0.06, false)
        } else if fingers.y - cube.y > 0.014 {
            (cube, false)
        } else {
            (cube, true)
        };
        let [a, b, c] = toward(obs, ik(target, opening));
        [a, b, c, close as usize]
    }

    #[test]
    fn servos_hold_and_follow() {
        let mut env = Arenas::new(So101::new(&config(4, 10_000)), 4, 0);
        let mut buffers = Buffers::new(&env);
        env.reset(buffers.agents());
        let start = buffers.obs.clone();
        // Targets still: gravity sags the arm, but not far
        run(&mut env, &mut buffers, 100, |_| [2, 2, 2, 0]);
        for (a, b) in start.chunks(28).zip(buffers.obs.chunks(28)) {
            let (qa, qb) = (joints(a), joints(b));
            for i in 0..3 {
                assert!((qa[i] - qb[i]).abs() < 0.08, "joint {i} sagged from {} to {}", qa[i], qb[i]);
            }
            assert!(b[9] > 0.9, "the gripper opened to {}", b[9] * OPEN_MAX);
        }
        // Up on every joint, a second: each goes up; and the gripper closes on nothing
        run(&mut env, &mut buffers, 50, |_| [3, 3, 3, 1]);
        for b in buffers.obs.chunks(28) {
            let q = joints(b);
            println!("q {q:?} opening {}", b[9] * OPEN_MAX);
            assert!(b[9] < 0.05, "the gripper closed to {}", b[9] * OPEN_MAX);
        }
    }

    #[test]
    fn ik_reaches() {
        let mut env = Arenas::new(So101::new(&config(4, 10_000)), 4, 1);
        let mut buffers = Buffers::new(&env);
        env.reset(buffers.agents());
        let to = Vec2::new(0.2, 0.1);
        run(&mut env, &mut buffers, 150, |obs| {
            let [a, b, c] = toward(obs, ik(to, obs[9] * OPEN_MAX));
            [a, b, c, 0]
        });
        for b in buffers.obs.chunks(28) {
            let fingers = Vec2::new(b[11], b[12]) * REACH.1;
            assert!(fingers.distance(to) < 0.01, "fingers at {fingers}, not {to}");
            assert!((b[14] + 1.0).abs() < 0.05, "hand at sin {} not straight down", b[14]);
        }
    }

    #[test]
    fn scripted_picks_and_places() {
        let n = 32;
        let mut env = Arenas::new(So101::new(&config(n, 500)), n, 5);
        let mut buffers = Buffers::new(&env);
        env.reset(buffers.agents());
        let total = run(&mut env, &mut buffers, 1000, scripted);
        let log = env.log();
        let get = |key: &str| log.sums.iter().find(|(k, _)| *k == key).map_or(0.0, |(_, v)| *v) / log.n as f64;
        println!("episodes {} goals {} perf {} grasped {} dropped {} reward {total}", log.n, get("score"), get("perf"), get("grasped"), get("dropped"));
        assert!(get("grasped") > 0.5);
        assert!(get("score") > 1.0);
    }
}
