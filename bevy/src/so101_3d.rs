//! SO-101 in three dimensions: the SO-ARM101 with all six of its servos, picking a cube up
//! off a table and carrying it to goals in the air, one after another.
//!
//! The arm is the real one's chain at its URDF's sizes, weights and joint limits: shoulder
//! pan on a 6 cm base, shoulder lift 11.7 cm above the table, an 11.6 cm upper arm, a
//! 13.5 cm forearm, wrist flex, then wrist roll and a hand 16 cm from wrist flex to
//! fingertips. Each joint is an STS3215-like position servo, at most 3 N·m, with its
//! geared rotor's inertia, which the policy steers by moving its target. The gripper is
//! the hand's flat face and a jaw that slides against it, a parallel stand-in for the
//! real arm's swinging jaw, pushed shut or open with 1.5 N as a current-limited servo
//! pushes: the cube is held by squeeze and friction alone. Wrist roll is what turns the
//! jaws to the cube's faces.
//!
//! Each episode the cube starts anywhere on the table in reach, turned any way about the
//! vertical, and a goal hangs in the air. Carrying the cube, held, to within 2 cm of the
//! goal is +5 and a new goal appears; max_steps (50 a second) ends the episode. Knocking
//! the cube off the table ends it too, for -2. On the way a step pays, as a change of
//! potential, for bringing the fingers nearer the cube, for holding it, and for bringing
//! it nearer the goal.
//!
//! Observations: joint angles, speeds and how far each servo's target is from its angle,
//! the gripper's opening and command, the fingers' place and the hand's two axes, the
//! cube's place, two axes and velocity, the goal, the cube from the fingers and the goal
//! from the cube, and whether the cube is held. Places are from the base's foot, y up and
//! the arm's zero pointing along x.
//!
//! Actions: six heads. Pan, lift, elbow, wrist flex and wrist roll each move their servo's
//! target 0 fast down, 1 slowly down, 2 not, 3 slowly up or 4 fast up; the gripper 0
//! opens, 1 closes.

use std::f32::consts::PI;

use puffer_bevy::bevy::app::App;
use puffer_bevy::bevy::color::Color;
use puffer_bevy::bevy::ecs::entity::Entity;
use puffer_bevy::bevy::ecs::world::World;
use puffer_bevy::bevy::math::{Quat, Vec3};
use puffer_bevy::three::avian3d::dynamics::solver::SolverConfig;
use puffer_bevy::three::avian3d::prelude::*;
use puffer_bevy::three::{placed, teleport, BevyEnv3d};
use puffer_bevy::view3::{Mark, Tint};
use puffer_bevy::Rng;
use pufferl::env::{Config, Log};

/// Shoulder pan's joint above the table, and shoulder lift's axis
const BASE_H: f32 = 0.0624;
const SHOULDER_H: f32 = 0.1166;
const UPPER: f32 = 0.116;
const FORE: f32 = 0.135;
const LINK_W: f32 = 0.026;
/// Wrist flex to wrist roll, and wrist roll to the fingertips
const WRIST: f32 = 0.06;
const HAND: f32 = 0.10;
/// The hand's thickness behind its gripping face, and its width
const HAND_W: f32 = 0.03;
const HAND_D: f32 = 0.03;
const JAW: f32 = 0.055;
const JAW_W: f32 = 0.01;
const OPEN_MAX: f32 = 0.06;
const CUBE: f32 = 0.032;
/// The grip point, wrist roll to between the jaws a cube's half and a little from the tips
const GRIP: f32 = HAND - CUBE / 2.0 - 0.005;

/// Masses from the URDF. The jaw is the real one's 12 g and its slide's.
const SHOULDER_KG: f32 = 0.100;
const UPPER_KG: f32 = 0.103;
const FORE_KG: f32 = 0.104;
const WRIST_KG: f32 = 0.079;
const HAND_KG: f32 = 0.087;
const JAW_KG: f32 = 0.04;
const CUBE_KG: f32 = 0.03;

/// The URDF's limits: pan, lift, elbow, wrist flex, wrist roll. Zero is the upper arm
/// straight up, the forearm square to it pointing along x, the hand in line with it and
/// its jaw opening upward.
const LIMITS: [f32; 5] = [1.92, 1.745, 1.69, 1.658, 2.74];
const MAX_TORQUE: f32 = 3.0;
/// Avian's acceleration-based joint motor: its stiffness, and the part of the speed error
/// it takes out each substep. The servo is also told its target's speed, so a moving
/// target isn't held back. 24 substeps let a stiff servo stay steady.
const SERVO_K: f32 = 1000000.0;
const SERVO_D: f32 = 0.3;
const SUBSTEPS: u32 = 24;
/// A geared servo's rotor, as its joint feels it: about 1e-7 kg·m² times the STS3215's
/// 345:1 gearing squared, more than the links themselves. Each link carries it about its
/// joint's axis, and a little of it about every axis (the servo's case), which is what
/// keeps a chain of Avian joint motors, each moving only the two light links it joins,
/// from ringing.
const ARMATURE: f32 = 0.003;
const ARMATURE_ALL: f32 = 0.001;
/// Each servo also damps its joint's speed at this rate a second, inside the substeps
const SERVO_DAMPING: f32 = 20.0;
/// The gripper pushes its jaw with this, as a current-limited servo does, and its slide
/// damps the jaw's speed at this rate a second: shutting or opening at up to about 1 m/s
const GRIP_FORCE: f32 = 1.5;
const GRIP_DAMPING: f32 = 50.0;
/// Of the jaws and the cube: higher, and a closing jaw shoves the cube across the table
/// rather than sliding to it
const GRIP_FRICTION: f32 = 1.0;
/// Target speeds, rad/s, fast and slow
const FAST: f32 = 2.5;
const SLOW: f32 = 0.6;
/// A servo's target stays this near its angle
const LEAD: f32 = 0.25;

/// Where cubes and goals are: this far from the base, this far to either side of
/// straight ahead, goals this high
const REACH: (f32, f32) = (0.12, 0.30);
const SPREAD: f32 = 1.0;
const GOAL_Y: (f32, f32) = (0.05, 0.20);
const GOAL_RADIUS: f32 = 0.02;
/// The table's top, x and z
const TABLE: (f32, f32, f32) = (-0.15, 0.45, 0.40);

const REACH_SHAPING: f32 = 10.0;
const HELD_SHAPING: f32 = 1.0;
const CARRY_SHAPING: f32 = 10.0;
const GOAL: f32 = 5.0;
const DROPPED: f32 = -2.0;

/// Contacts stiffer than Avian's default, or the jaw sinks into the cube and it slips out
const CONTACT_HZ: f32 = 4.0;
const OVERLAP_SPEED: f32 = 40.0;

const L_TABLE: u32 = 1;
const L_ARM: u32 = 2;
const L_BASE: u32 = 4;
const L_CUBE: u32 = 8;

const ARM_COLOR: Color = Color::srgb(1.0, 0.78, 0.15);
const CUBE_COLOR: Color = Color::srgb(0.9, 0.25, 0.3);

/// The arm's links, base to jaw: the shoulder turns on the base, and each after on the
/// one before
const SHOULDER: usize = 0;
const UPPER_ARM: usize = 1;
const FOREARM: usize = 2;
const WRIST_LINK: usize = 3;
const HAND_LINK: usize = 4;
const JAW_LINK: usize = 5;

/// A link at the zero pose: its center and size, its mass
struct Link {
    center: Vec3,
    size: Vec3,
    kg: f32,
}

fn links() -> [Link; 6] {
    let elbow = Vec3::new(0.0, SHOULDER_H + UPPER, 0.0);
    let wrist = elbow + Vec3::X * FORE;
    let roll = wrist + Vec3::X * WRIST;
    [
        Link { center: Vec3::new(0.0, (BASE_H + SHOULDER_H) / 2.0, 0.0), size: Vec3::new(0.04, SHOULDER_H - BASE_H, 0.05), kg: SHOULDER_KG },
        Link { center: Vec3::new(0.0, SHOULDER_H + UPPER / 2.0, 0.0), size: Vec3::new(LINK_W, UPPER, LINK_W), kg: UPPER_KG },
        Link { center: elbow + Vec3::X * FORE / 2.0, size: Vec3::new(FORE, LINK_W, LINK_W), kg: FORE_KG },
        Link { center: wrist + Vec3::X * WRIST / 2.0, size: Vec3::new(WRIST, 0.03, 0.035), kg: WRIST_KG },
        Link { center: roll + Vec3::new(HAND / 2.0, -HAND_W / 2.0, 0.0), size: Vec3::new(HAND, HAND_W, HAND_D), kg: HAND_KG },
        // Shut: its inner face on the hand's
        Link { center: roll + Vec3::new(HAND - JAW / 2.0, JAW_W / 2.0, 0.0), size: Vec3::new(JAW, JAW_W, HAND_D), kg: JAW_KG },
    ]
}

/// The five servos at the zero pose: the point each turns about, and its axis
fn pivots() -> [(Vec3, Vec3); 5] {
    let elbow = Vec3::new(0.0, SHOULDER_H + UPPER, 0.0);
    [
        (Vec3::new(0.0, BASE_H, 0.0), Vec3::Y),
        (Vec3::new(0.0, SHOULDER_H, 0.0), Vec3::Z),
        (elbow, Vec3::Z),
        (elbow + Vec3::X * FORE, Vec3::Z),
        (elbow + Vec3::X * (FORE + WRIST), Vec3::X),
    ]
}

/// The wrist roll's point at the zero pose: the hand's frame starts there
fn roll_point() -> Vec3 {
    pivots()[4].0
}

/// A rigid motion: points at the zero pose to where they are now
#[derive(Clone, Copy)]
struct Motion {
    rotation: Quat,
    shift: Vec3,
}

impl Motion {
    const ID: Self = Self { rotation: Quat::IDENTITY, shift: Vec3::ZERO };

    fn apply(&self, p: Vec3) -> Vec3 {
        self.rotation * p + self.shift
    }

    /// self, then other's motion: other is about the zero pose's points
    fn then(&self, other: Motion) -> Self {
        Self { rotation: self.rotation * other.rotation, shift: self.rotation * other.shift + self.shift }
    }

    fn turn_about(point: Vec3, axis: Vec3, angle: f32) -> Self {
        let rotation = Quat::from_axis_angle(axis, angle);
        Self { rotation, shift: point - rotation * point }
    }
}

/// Every link's motion for joint angles q and a gripper opening
fn kinematics(q: [f32; 5], opening: f32) -> [Motion; 6] {
    let pivots = pivots();
    let mut motions = [Motion::ID; 6];
    let mut motion = Motion::ID;
    for (i, &(point, axis)) in pivots.iter().enumerate() {
        motion = motion.then(Motion::turn_about(point, axis, q[i]));
        motions[i] = motion;
    }
    motions[JAW_LINK] = motion.then(Motion { rotation: Quat::IDENTITY, shift: Vec3::Y * opening });
    motions
}

/// The angle of rotation about axis
fn angle_about(rotation: Quat, axis: Vec3) -> f32 {
    let r = if rotation.w < 0.0 { -rotation } else { rotation };
    wrap(2.0 * Vec3::new(r.x, r.y, r.z).dot(axis).atan2(r.w))
}

fn wrap(a: f32) -> f32 {
    (a + PI).rem_euclid(2.0 * PI) - PI
}

pub struct So101x3 {
    max_steps: usize,
}

impl So101x3 {
    pub fn new(config: &Config) -> Self {
        Self { max_steps: config.size("max_steps") }
    }
}

pub struct Cell {
    /// shoulder, upper arm, forearm, wrist, hand, jaw
    links: [Entity; 6],
    cube: Entity,
    joints: [Entity; 5],
    targets: [f32; 5],
    close: bool,
    goal: Vec3,
    potential: f32,
    goals: u32,
    ever_held: bool,
    dropped: bool,
    rng: Rng,
}

/// What the arm knows of itself, read off its bodies
struct Pose {
    q: [f32; 5],
    qd: [f32; 5],
    /// The hand's motion from the zero pose, and the gripper's opening
    hand: Motion,
    opening: f32,
}

fn body(world: &World, entity: Entity) -> (Vec3, Quat) {
    (world.get::<Position>(entity).unwrap().0, world.get::<Rotation>(entity).unwrap().0)
}

fn spin(world: &World, entity: Entity) -> Vec3 {
    world.get::<AngularVelocity>(entity).unwrap().0
}

impl Pose {
    fn read(world: &World, cell: &Cell, origin: Vec3) -> Self {
        let links = links();
        let pivots = pivots();
        let motion = |i: usize| {
            let (at, rotation) = body(world, cell.links[i]);
            Motion { rotation, shift: at - origin - rotation * links[i].center }
        };
        let mut q = [0.0; 5];
        let mut qd = [0.0; 5];
        let mut parent = (Quat::IDENTITY, Vec3::ZERO);
        for i in 0..5 {
            let (_, rotation) = body(world, cell.links[i]);
            let w = spin(world, cell.links[i]);
            let axis = parent.0 * pivots[i].1;
            q[i] = angle_about(parent.0.inverse() * rotation, pivots[i].1);
            qd[i] = (w - parent.1).dot(axis);
            parent = (rotation, w);
        }
        let hand = motion(HAND_LINK);
        let jaw = motion(JAW_LINK);
        let opening = (hand.rotation.inverse() * (jaw.shift - hand.shift)).y;
        Pose { q, qd, hand, opening }
    }

    /// Where between the jaws a cube is gripped
    fn fingers(&self) -> Vec3 {
        self.hand.apply(roll_point() + Vec3::new(GRIP, self.opening / 2.0, 0.0))
    }

    /// A place in the hand's frame, from the wrist roll's point
    fn in_hand(&self, at: Vec3) -> Vec3 {
        self.hand.rotation.inverse() * (at - self.hand.apply(roll_point()))
    }
}

/// The cube is held when it's between the jaws, they're closing, and they're on it: open
/// as wide as the cube is across them, turned as it is in the hand
fn held(pose: &Pose, cube: Vec3, cube_rotation: Quat, close: bool) -> bool {
    let c = pose.in_hand(cube);
    let across_axis = pose.hand.rotation * Vec3::Y;
    let across = CUBE * [Vec3::X, Vec3::Y, Vec3::Z].iter().map(|&a| (cube_rotation * a).dot(across_axis).abs()).sum::<f32>();
    close
        && (HAND - JAW - CUBE * 0.2..=HAND + CUBE * 0.2).contains(&c.x)
        && (0.0..=pose.opening).contains(&c.y)
        && c.z.abs() <= HAND_D / 2.0 + CUBE * 0.25
        && (across - 0.006..=across + 0.003).contains(&pose.opening)
}

/// A place in reach: distance from the base and turn from straight ahead
fn in_reach(rng: &mut Rng, y: f32) -> Vec3 {
    let (r, turn) = (rng.range(REACH.0, REACH.1), rng.range(-SPREAD, SPREAD));
    Vec3::new(r * turn.cos(), y, -r * turn.sin())
}

impl So101x3 {
    fn potential(&self, world: &World, cell: &Cell, origin: Vec3) -> (f32, bool) {
        let pose = Pose::read(world, cell, origin);
        let (cube, cube_rotation) = body(world, cell.cube);
        let cube = cube - origin;
        let holding = held(&pose, cube, cube_rotation, cell.close);
        let phi = -REACH_SHAPING * pose.fingers().distance(cube) + if holding { HELD_SHAPING } else { 0.0 }
            - CARRY_SHAPING * cube.distance(cell.goal);
        (phi, holding)
    }

    fn new_goal(rng: &mut Rng, cube: Vec3) -> Vec3 {
        let mut goal = Vec3::ZERO;
        for _ in 0..100 {
            let y = rng.range(GOAL_Y.0, GOAL_Y.1);
            goal = in_reach(rng, y);
            if goal.distance(cube) > 0.06 {
                break;
            }
        }
        goal
    }
}

impl BevyEnv3d for So101x3 {
    type Arena = Cell;

    fn obs_size(&self) -> usize {
        48
    }

    fn act_sizes(&self) -> Vec<usize> {
        vec![5, 5, 5, 5, 5, 2]
    }

    fn spacing(&self) -> f32 {
        1.0
    }

    fn build(&self, app: &mut App) {
        // The arm is centimeters across, not meters: Avian's tolerances scale with this
        app.insert_resource(PhysicsLengthUnit(0.05));
        app.insert_resource(SubstepCount(SUBSTEPS));
        app.insert_resource(SolverConfig { contact_frequency_factor: CONTACT_HZ, max_overlap_solve_speed: OVERLAP_SPEED, ..Default::default() });
    }

    /// The goal: green while the cube is held, yellow while not
    fn marks(&self, world: &World, cell: &Cell, origin: Vec3, marks: &mut Vec<Mark>) {
        let held_now = self.potential(world, cell, origin).1;
        let color = if held_now { Color::srgb(0.2, 0.85, 0.4) } else { Color::srgb(0.9, 0.8, 0.2) };
        marks.push(Mark::Sphere { at: origin + cell.goal, radius: GOAL_RADIUS, color });
    }

    fn spawn(&self, world: &mut World, origin: Vec3) -> Cell {
        let layers = |member: u32, filter: u32| CollisionLayers::from_bits(member, filter);
        let (x0, x1, half_z) = TABLE;
        world.spawn((
            RigidBody::Static,
            Collider::cuboid(x1 - x0, 0.04, 2.0 * half_z),
            layers(L_TABLE, L_ARM | L_CUBE),
            Friction::new(0.6),
            placed(origin + Vec3::new((x0 + x1) / 2.0, -0.02, 0.0), Quat::IDENTITY),
        ));
        let base = world
            .spawn((
                RigidBody::Static,
                Collider::cuboid(0.07, BASE_H, 0.07),
                layers(L_BASE, L_CUBE),
                Tint(ARM_COLOR),
                placed(origin + Vec3::Y * BASE_H / 2.0, Quat::IDENTITY),
            ))
            .id();

        let parts = links();
        let links: [Entity; 6] = std::array::from_fn(|i| {
            let Link { center, size, kg } = parts[i];
            // A box's inertia, and the servo's geared rotor about the joint's axis
            let sq = size * size;
            let armature = if i < 5 { pivots()[i].1 * ARMATURE + Vec3::splat(ARMATURE_ALL) } else { Vec3::ZERO };
            let inertia = Vec3::new(sq.y + sq.z, sq.x + sq.z, sq.x + sq.y) * kg / 12.0 + armature;
            let mut body = world.spawn((
                RigidBody::Dynamic,
                Collider::cuboid(size.x, size.y, size.z),
                ColliderDensity(0.0),
                Mass(kg),
                AngularInertia::new(inertia),
                layers(L_ARM, L_TABLE | L_CUBE),
                Friction::new(GRIP_FRICTION),
                Restitution::new(0.0),
                SleepingDisabled,
                Tint(ARM_COLOR),
                placed(origin + center, Quat::IDENTITY),
            ));
            if i >= HAND_LINK {
                body.insert(ConstantForce::new(0.0, 0.0, 0.0));
            }
            body.id()
        });
        let cube = world
            .spawn((
                RigidBody::Dynamic,
                Collider::cuboid(CUBE, CUBE, CUBE),
                ColliderDensity(CUBE_KG / CUBE.powi(3)),
                layers(L_CUBE, L_TABLE | L_ARM | L_BASE),
                Friction::new(GRIP_FRICTION),
                Restitution::new(0.0),
                SleepingDisabled,
                Tint(CUBE_COLOR),
                placed(origin + Vec3::new(0.2, CUBE / 2.0, 0.0), Quat::IDENTITY),
            ))
            .id();

        // Every body is unturned at the zero pose, so each joint's frames are the pivot
        // from each body's center and no turn: its angle is the turn from the zero pose
        let motor = || AngularMotor::new(MotorModel::AccelerationBased { stiffness: SERVO_K, damping: SERVO_D }).with_max_torque(MAX_TORQUE);
        let parents = [base, links[SHOULDER], links[UPPER_ARM], links[FOREARM], links[WRIST_LINK]];
        let base_center = Vec3::Y * BASE_H / 2.0;
        let joints: [Entity; 5] = std::array::from_fn(|i| {
            let (point, axis) = pivots()[i];
            let parent_center = if i == 0 { base_center } else { parts[i - 1].center };
            world
                .spawn((
                    RevoluteJoint::new(parents[i], links[i])
                        .with_hinge_axis(axis)
                        .with_local_anchor1(point - parent_center)
                        .with_local_anchor2(point - parts[i].center)
                        .with_local_basis1(Quat::IDENTITY)
                        .with_local_basis2(Quat::IDENTITY)
                        .with_angle_limits(-LIMITS[i], LIMITS[i])
                        .with_motor(motor()),
                    JointDamping { linear: 0.0, angular: SERVO_DAMPING },
                    JointCollisionDisabled,
                ))
                .id()
        });
        // The jaw slides square to the hand's face, its inner side opening from it
        world.spawn((
            PrismaticJoint::new(links[HAND_LINK], links[JAW_LINK])
                .with_slider_axis(Vec3::Y)
                .with_local_anchor1(parts[JAW_LINK].center - parts[HAND_LINK].center)
                .with_local_anchor2(Vec3::ZERO)
                .with_local_basis1(Quat::IDENTITY)
                .with_local_basis2(Quat::IDENTITY)
                .with_limits(0.0, OPEN_MAX),
            JointDamping { linear: GRIP_DAMPING, angular: 0.0 },
            JointCollisionDisabled,
        ));

        Cell {
            links,
            cube,
            joints,
            targets: [0.0; 5],
            close: false,
            goal: Vec3::new(0.2, 0.1, 0.0),
            potential: 0.0,
            goals: 0,
            ever_held: false,
            dropped: false,
            rng: Rng::new(0),
        }
    }

    /// The arm near its rest pose, the gripper some way open, the cube anywhere on the
    /// table in reach and turned any way, and a goal in the air
    fn reset(&self, world: &mut World, cell: &mut Cell, origin: Vec3, rng: &mut Rng) {
        let q = [rng.range(-0.3, 0.3), rng.range(-0.2, 0.2), rng.range(0.0, 0.4), rng.range(-0.4, 0.2), rng.range(-0.3, 0.3)];
        let opening = rng.range(0.0, OPEN_MAX);
        let motions = kinematics(q, opening);
        let parts = links();
        for i in 0..6 {
            teleport(world, cell.links[i], origin + motions[i].apply(parts[i].center), motions[i].rotation, Vec3::ZERO, Vec3::ZERO);
        }
        let cube = in_reach(rng, CUBE / 2.0);
        let yaw = Quat::from_rotation_y(rng.range(0.0, PI / 2.0));
        teleport(world, cell.cube, origin + cube, yaw, Vec3::ZERO, Vec3::ZERO);

        for (joint, &target) in cell.joints.iter().zip(&q) {
            let motor = &mut world.get_mut::<RevoluteJoint>(*joint).unwrap().motor;
            (motor.target_position, motor.target_velocity) = (target, 0.0);
        }
        for link in [HAND_LINK, JAW_LINK] {
            world.get_mut::<ConstantForce>(cell.links[link]).unwrap().0 = Vec3::ZERO;
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
        let pose = Pose::read(world, cell, Vec3::ZERO);
        for i in 0..5 {
            let speed = [-FAST, -SLOW, 0.0, SLOW, FAST][actions[i]];
            let target = (cell.targets[i] + speed * dt).clamp(pose.q[i] - LEAD, pose.q[i] + LEAD).clamp(-LIMITS[i], LIMITS[i]);
            // The servo follows its target's speed too, so a moving target isn't damped
            let motor = &mut world.get_mut::<RevoluteJoint>(cell.joints[i]).unwrap().motor;
            (motor.target_position, motor.target_velocity) = (target, (target - cell.targets[i]) / dt);
            cell.targets[i] = target;
        }
        cell.close = actions[5] == 1;
        // Along the jaw's slide: open is the hand's +y
        let slide = pose.hand.rotation * Vec3::Y;
        let push = if cell.close { -GRIP_FORCE } else { GRIP_FORCE };
        world.get_mut::<ConstantForce>(cell.links[JAW_LINK]).unwrap().0 = slide * push;
        world.get_mut::<ConstantForce>(cell.links[HAND_LINK]).unwrap().0 = -slide * push;
    }

    fn observe(&self, world: &World, cell: &Cell, origin: Vec3, obs: &mut [f32]) {
        let pose = Pose::read(world, cell, origin);
        let (cube, cube_rotation) = body(world, cell.cube);
        let cube = cube - origin;
        let cube_v = world.get::<LinearVelocity>(cell.cube).unwrap().0;
        let fingers = pose.fingers();
        let holding = held(&pose, cube, cube_rotation, cell.close);
        let s = 1.0 / REACH.1;
        let clamp = |v: Vec3| v.clamp(Vec3::splat(-3.0), Vec3::splat(3.0));
        let mut o = Vec::with_capacity(48);
        for i in 0..5 {
            o.push(pose.q[i] / LIMITS[i]);
        }
        for i in 0..5 {
            o.push(pose.qd[i] / 5.0);
        }
        for i in 0..5 {
            o.push((cell.targets[i] - pose.q[i]) / LEAD);
        }
        o.extend([pose.opening / OPEN_MAX, if cell.close { 1.0 } else { 0.0 }]);
        for v in [
            fingers * s,
            pose.hand.rotation * Vec3::X,
            pose.hand.rotation * Vec3::Y,
            cube * s,
            cube_rotation * Vec3::X,
            cube_rotation * Vec3::Y,
            cube_v,
            cell.goal * s,
            clamp((cube - fingers) * 10.0),
            clamp((cell.goal - cube) * 10.0),
        ] {
            o.extend(v.to_array());
        }
        o.push(if holding { 1.0 } else { 0.0 });
        obs.copy_from_slice(&o);
    }

    fn outcome(&self, world: &World, cell: &mut Cell, origin: Vec3, tick: usize) -> (f32, bool) {
        let (phi, holding) = self.potential(world, cell, origin);
        let mut reward = phi - cell.potential;
        cell.potential = phi;
        cell.ever_held |= holding;
        let cube = body(world, cell.cube).0 - origin;
        if holding && cube.distance(cell.goal) < GOAL_RADIUS {
            reward += GOAL;
            cell.goals += 1;
            // The next goal, with a fresh potential so its distance isn't a penalty
            cell.goal = Self::new_goal(&mut cell.rng, cube);
            cell.potential = self.potential(world, cell, origin).0;
        }
        let (x0, x1, half_z) = TABLE;
        // Off the table, or a world gone wrong: either ends the episode
        let arm_ok = cell.links.iter().all(|&link| body(world, link).0.is_finite());
        if !cube.is_finite() || !arm_ok || cube.y < -0.01 || !(x0..x1).contains(&cube.x) || cube.z.abs() > half_z {
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
    use puffer_bevy::bevy::ecs::world::World;
    use puffer_bevy::three::Arenas3d;
    use pufferl::env::Env;

    const N: usize = 48;
    const HEADS: usize = 6;

    fn run(env: &mut Arenas3d<So101x3>, buffers: &mut Buffers, n: usize, policy: impl Fn(&[f32]) -> [usize; HEADS]) {
        for _ in 0..n {
            for (actions, obs) in buffers.actions.chunks_mut(HEADS).zip(buffers.obs.chunks(N)) {
                for (a, b) in actions.iter_mut().zip(policy(obs)) {
                    *a = b as f32;
                }
            }
            env.step(buffers.agents());
        }
    }

    fn joints(obs: &[f32]) -> [f32; 5] {
        std::array::from_fn(|i| obs[i] * LIMITS[i])
    }

    fn v3(obs: &[f32], at: usize) -> Vec3 {
        Vec3::new(obs[at], obs[at + 1], obs[at + 2])
    }

    #[test]
    fn kinematics_round_trip() {
        // A pose put in place reads back as the same joint angles and opening
        let mut env = Arenas3d::new(So101x3::new(&config(8, 1000)), 8, 2);
        let mut buffers = Buffers::new(&env);
        env.reset(buffers.agents());
        let world = env.world();
        let _ = world;
        for obs in buffers.obs.chunks(N) {
            let q = joints(obs);
            let fingers = v3(obs, 17) * REACH.1;
            let expect = kinematics(q, obs[15] * OPEN_MAX)[HAND_LINK].apply(roll_point() + Vec3::new(GRIP, obs[15] * OPEN_MAX / 2.0, 0.0));
            assert!(fingers.distance(expect) < 1e-4, "fingers {fingers} vs kinematics {expect}");
        }
    }

    #[test]
    fn servos_hold_and_follow() {
        let mut env = Arenas3d::new(So101x3::new(&config(4, 10_000)), 4, 0);
        let mut buffers = Buffers::new(&env);
        env.reset(buffers.agents());
        let start = buffers.obs.clone();
        run(&mut env, &mut buffers, 100, |_| [2, 2, 2, 2, 2, 0]);
        for (a, b) in start.chunks(N).zip(buffers.obs.chunks(N)) {
            let (qa, qb) = (joints(a), joints(b));
            for i in 0..5 {
                assert!((qa[i] - qb[i]).abs() < 0.08, "joint {i} went from {} to {}", qa[i], qb[i]);
            }
            assert!(b[15] > 0.9, "the gripper opened to {}", b[15] * OPEN_MAX);
        }
        // Every joint up for a second: each follows its target
        run(&mut env, &mut buffers, 50, |_| [3, 3, 3, 3, 3, 1]);
        for (a, b) in start.chunks(N).zip(buffers.obs.chunks(N)) {
            let (qa, qb) = (joints(a), joints(b));
            for i in 0..5 {
                assert!(qb[i] - qa[i] > 0.3, "joint {i} moved only from {} to {}", qa[i], qb[i]);
            }
            assert!(b[15] < 0.05, "the gripper closed to {}", b[15] * OPEN_MAX);
        }
    }

    /// Joint angles that put the fingers, the gripper open by opening, on target with the
    /// hand pointing down (tilted forward as far as the wrist needs) and rolled by roll.
    /// Pointing down and rolled, the jaw opens along R_y(pan - roll) x.
    fn ik(target: Vec3, roll: f32, opening: f32) -> [f32; 5] {
        // Aim the hand's axis, then correct by where the fingers really end up
        let mut aim = target;
        let mut q = planar(aim, roll);
        for _ in 0..4 {
            let fingers = kinematics(q, opening)[HAND_LINK].apply(roll_point() + Vec3::new(GRIP, opening / 2.0, 0.0));
            aim += target - fingers;
            q = planar(aim, roll);
        }
        q
    }

    /// The hand straight down, or tilted forward as far as the joints need: the first hand
    /// pitch, from straight down, whose angles are all inside the limits. Straight down
    /// keeps the jaws' faces upright, which is what holds a cube.
    fn planar(target: Vec3, roll: f32) -> [f32; 5] {
        let pan = (-target.z).atan2(target.x);
        let (r, y) = (Vec3::new(target.x, 0.0, target.z).length(), target.y);
        let reach = WRIST + GRIP;
        let mut best = [pan, 0.0, 0.0, 0.0, roll];
        for k in 0..60 {
            let a3 = -PI / 2.0 + k as f32 * 0.02;
            let wrist = (r - reach * a3.cos(), y - reach * a3.sin());
            let to = (wrist.0, wrist.1 - SHOULDER_H);
            let d = to.0.hypot(to.1).clamp(0.03, UPPER + FORE - 1e-3);
            let inner = ((UPPER * UPPER + d * d - FORE * FORE) / (2.0 * UPPER * d)).clamp(-1.0, 1.0).acos();
            let a1 = to.1.atan2(to.0) + inner;
            let elbow = (UPPER * a1.cos(), SHOULDER_H + UPPER * a1.sin());
            let a2 = (wrist.1 - elbow.1).atan2(wrist.0 - elbow.0);
            best = [pan, wrap(a1 - PI / 2.0), wrap(a2 - a1 + PI / 2.0), wrap(a3 - a2), roll];
            if (1..4).all(|i| best[i].abs() < LIMITS[i] - 0.05) {
                break;
            }
        }
        best
    }

    fn toward(obs: &[f32], q: [f32; 5]) -> [usize; 5] {
        toward_at(obs, q, true)
    }

    /// Move each servo's target toward q, fast when far if fast, else slowly
    fn toward_at(obs: &[f32], q: [f32; 5], fast: bool) -> [usize; 5] {
        let now = joints(obs);
        std::array::from_fn(|i| {
            let e = q[i] - (now[i] + obs[10 + i] * LEAD);
            // A fast step is 0.05 rad, a slow one 0.012
            let (up, down) = if fast && e.abs() > 0.05 { (4, 0) } else { (3, 1) };
            if e > 0.006 { up } else if e < -0.006 { down } else { 2 }
        })
    }

    /// Over the cube with the jaws turned square to it, down, close, carry to the goal
    fn scripted(obs: &[f32]) -> [usize; HEADS] {
        let s = REACH.1;
        let (fingers, cube, goal) = (v3(obs, 17) * s, v3(obs, 26) * s, v3(obs, 38) * s);
        // The cube's most level axis: a face's normal, however the cube lies
        let (cx, cy) = (v3(obs, 29), v3(obs, 32));
        let cube_x = [cx, cy, cx.cross(cy)].into_iter().min_by(|a, b| a.y.abs().total_cmp(&b.y.abs())).unwrap();
        let opening = obs[15] * OPEN_MAX;
        let q = joints(obs);
        // Roll so the jaw opens along a face of the cube, and across the arm rather than
        // along it, so the hand's forward tilt leaves the jaws' faces upright: pan - roll
        // is the cube's turn, give or take quarter turns, with roll near a quarter turn
        let pan = (-cube.z).atan2(cube.x);
        let face = (-cube_x.z).atan2(cube_x.x);
        let roll = wrap((pan - face - PI / 2.0 + PI / 4.0).rem_euclid(PI / 2.0) - PI / 4.0 + PI / 2.0);
        let holding = obs[47] > 0.5;
        let closing = obs[16] > 0.5 && opening > CUBE * 0.8 && fingers.distance(cube) < 0.03;
        let flat_error = Vec3::new(cube.x - fingers.x, 0.0, cube.z - fingers.z).length();
        let descending = flat_error < 0.005 || (flat_error < 0.015 && fingers.y < cube.y + 0.07);
        let (target, close, fast) = if holding {
            (goal, true, false)
        } else if closing {
            (cube, true, false)
        } else if !descending || (q[4] - roll).abs() > 0.05 {
            (cube + Vec3::Y * 0.08, false, flat_error > 0.03)
        } else if fingers.y - cube.y > 0.014 {
            (cube, false, false)
        } else {
            (cube, true, false)
        };
        let [a, b, c, d, e] = toward_at(obs, ik(target, if holding { q[4] } else { roll }, opening), fast);
        [a, b, c, d, e, close as usize]
    }

    /// One arena in a world of its own, its cell at hand
    fn bare() -> (World, Cell, So101x3) {
        use puffer_bevy::bevy::app::{App, TaskPoolPlugin};
        use puffer_bevy::bevy::time::{Fixed, Time, TimePlugin, TimeUpdateStrategy};
        use puffer_bevy::bevy::transform::TransformPlugin;
        let env = So101x3::new(&config(1, 10_000));
        let mut app = App::new();
        app.add_plugins((TaskPoolPlugin::default(), TimePlugin, TransformPlugin, PhysicsPlugins::default()))
            .insert_resource(Time::<Fixed>::from_duration(std::time::Duration::from_secs_f32(0.02)))
            .insert_resource(TimeUpdateStrategy::FixedTimesteps(1));
        env.build(&mut app);
        app.finish();
        app.cleanup();
        let mut world = std::mem::take(app.world_mut());
        let mut cell = env.spawn(&mut world, Vec3::ZERO);
        env.reset(&mut world, &mut cell, Vec3::ZERO, &mut Rng::new(1));
        (world, cell, env)
    }

    fn step(world: &mut World) {
        world.run_schedule(puffer_bevy::bevy::app::Main);
        world.clear_trackers();
    }

    /// The hand straight down over a cube, the jaws shut on it, then up: the cube comes too,
    /// held by squeeze and friction alone
    #[test]
    fn grips_and_lifts() {
        let (mut world, mut cell, env) = bare();
        let at = Vec3::new(0.2, CUBE / 2.0 + 0.008, 0.0);
        let q = ik(at, PI / 2.0, OPEN_MAX);
        let motions = kinematics(q, OPEN_MAX);
        let parts = links();
        for i in 0..6 {
            teleport(&mut world, cell.links[i], motions[i].apply(parts[i].center), motions[i].rotation, Vec3::ZERO, Vec3::ZERO);
        }
        teleport(&mut world, cell.cube, Vec3::new(at.x, CUBE / 2.0, at.z), Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO);
        cell.targets = q;
        let mut obs = vec![0.0; 48];
        let mut lifted = false;
        for t in 0..100 {
            env.observe(&world, &cell, Vec3::ZERO, &mut obs);
            // Settle and close, then lift straight up
            let up = if t < 40 { at } else { at + Vec3::Y * 0.12 };
            let [j0, j1, j2, j3, j4] = toward_at(&obs, ik(up, PI / 2.0, obs[15] * OPEN_MAX), false);
            let a = [j0, j1, j2, j3, j4, 1];
            env.act(&mut world, &mut cell, &a);
            step(&mut world);
            let pose = Pose::read(&world, &cell, Vec3::ZERO);
            let (cube, rot) = body(&world, cell.cube);
            if t == 20 {
                assert!(held(&pose, cube, rot, cell.close), "not held once closed: opening {}", pose.opening);
            }
            lifted |= t > 40 && cube.y > 0.08 && held(&pose, cube, rot, cell.close);
        }
        assert!(lifted, "the cube never rose 8 cm held");
    }

    #[test]
    fn ik_reaches() {
        let mut env = Arenas3d::new(So101x3::new(&config(4, 10_000)), 4, 1);
        let mut buffers = Buffers::new(&env);
        env.reset(buffers.agents());
        let to = Vec3::new(0.15, 0.1, -0.1);
        run(&mut env, &mut buffers, 150, |obs| {
            let [a, b, c, d, e] = toward(obs, ik(to, 0.3, obs[15] * OPEN_MAX));
            [a, b, c, d, e, 0]
        });
        for obs in buffers.obs.chunks(N) {
            let fingers = v3(obs, 17) * REACH.1;
            assert!(fingers.distance(to) < 0.012, "fingers at {fingers}, not {to}");
            assert!(obs[21] < -0.95, "hand's axis {} is not pointing down", v3(obs, 20));
        }
    }

    #[test]
    fn scripted_picks_and_places() {
        let n = 32;
        let mut env = Arenas3d::new(So101x3::new(&config(n, 500)), n, 5);
        let mut buffers = Buffers::new(&env);
        env.reset(buffers.agents());
        run(&mut env, &mut buffers, 1000, scripted);
        let log = env.log();
        let get = |key: &str| log.sums.iter().find(|(k, _)| *k == key).map_or(0.0, |(_, v)| *v) / log.n as f64;
        println!("episodes {} goals {} perf {} grasped {} dropped {}", log.n, get("score"), get("perf"), get("grasped"), get("dropped"));
        // A crude script: it grabs the cube more often than not, and carries it now and then
        assert!(get("grasped") > 0.5);
        assert!(get("dropped") < 0.2);
    }
}
