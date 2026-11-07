//! Watching a policy play: a Bevy window on the main thread, eval on a thread behind it.
//!
//! [`run`] opens the window and starts eval on a thread of its own. Eval calls an env's
//! render every step; [`Arenas`](crate::Arenas) answers with a [`Frame`], every body in its
//! world as its collider's shape where it is now, and [`show`]s it. The window draws the
//! latest frame each time it redraws. Closing the window ends eval (window_closed), and eval
//! ending closes the window.
//!
//! Bodies are drawn from their colliders, so an env is drawn with no drawing code of its
//! own; what isn't a body, a goal or a track, an env adds as [`Mark`]s.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use bevy::camera::ScalingMode;
use bevy::prelude::*;

/// A body's shape, as its collider has it
#[derive(Clone, Copy)]
pub enum Outline {
    Circle(f32),
    /// Half the width and half the height
    Rect(Vec2),
}

#[derive(Clone)]
pub struct Body {
    pub at: Vec2,
    pub angle: f32,
    pub outline: Outline,
    pub dynamic: bool,
}

/// Something drawn that is not a body
#[derive(Clone)]
pub enum Mark {
    Circle { at: Vec2, radius: f32, color: Color },
    Line { from: Vec2, to: Vec2, color: Color },
}

/// The world as it is after a step
#[derive(Clone, Default)]
pub struct Frame {
    pub bodies: Vec<Body>,
    pub marks: Vec<Mark>,
}

#[derive(Default)]
struct Shared {
    frame: Mutex<Option<Frame>>,
    closed: AtomicBool,
    eval_done: AtomicBool,
}

static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();

/// Whether a window is watching, so render has someone to draw for
pub fn watching() -> bool {
    SHARED.get().is_some()
}

/// The frame for the window to draw next
pub fn show(frame: Frame) {
    if let Some(shared) = SHARED.get() {
        *shared.frame.lock().unwrap() = Some(frame);
    }
}

/// Whether the window has been closed
pub fn closed() -> bool {
    SHARED.get().is_some_and(|shared| shared.closed.load(Ordering::Relaxed))
}

/// Open the window, and run eval behind it until one or the other ends
pub fn run(eval: impl FnOnce() + Send + 'static) {
    let shared = SHARED.get_or_init(Default::default).clone();
    let behind = shared.clone();
    let eval = std::thread::Builder::new()
        .name("eval".into())
        .spawn(move || {
            eval();
            behind.eval_done.store(true, Ordering::Relaxed);
        })
        .expect("a thread for eval");

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { title: "puff-rs · bevy".into(), ..default() }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.06, 0.065, 0.08)))
        .insert_resource(Link(shared.clone()))
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(Camera2d);
        })
        .add_systems(Update, (draw, end_with_eval))
        .run();
    shared.closed.store(true, Ordering::Relaxed);
    let _ = eval.join();
}

/// The camera, apart from the drawn bodies
type CameraQuery<'w, 's> = Query<'w, 's, (&'static mut Transform, &'static mut Projection), (With<Camera2d>, Without<Piece>)>;

#[derive(Resource)]
struct Link(Arc<Shared>);

/// The drawn body for frame.bodies[.0]
#[derive(Component)]
struct Piece(usize);

const STATIC: Color = Color::srgb(0.32, 0.34, 0.40);
const ROUND: Color = Color::srgb(1.0, 0.53, 0.0);
const SQUARE: Color = Color::srgb(0.30, 0.62, 1.0);

#[allow(clippy::too_many_arguments)]
fn draw(
    link: Res<Link>,
    mut last: Local<Option<Frame>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut pieces: Query<(Entity, &Piece, &mut Transform)>,
    mut camera: CameraQuery,
    mut gizmos: Gizmos,
) {
    if let Some(frame) = link.0.frame.lock().unwrap().take() {
        let respawn = last.as_ref().is_none_or(|last| last.bodies.len() != frame.bodies.len());
        *last = Some(frame);
        let frame = last.as_ref().unwrap();
        if respawn {
            for (entity, _, _) in &pieces {
                commands.entity(entity).despawn();
            }
            let (color_static, color_round, color_square) =
                (materials.add(STATIC), materials.add(ROUND), materials.add(SQUARE));
            for (i, body) in frame.bodies.iter().enumerate() {
                let (mesh, color) = match body.outline {
                    Outline::Circle(r) => (meshes.add(Circle::new(r)), &color_round),
                    Outline::Rect(half) => (meshes.add(Rectangle::from_size(half * 2.0)), &color_square),
                };
                let color = if body.dynamic { color.clone() } else { color_static.clone() };
                commands.spawn((Mesh2d(mesh), MeshMaterial2d(color), place(body), Piece(i)));
            }
            fit(frame, &mut camera);
        } else {
            for (_, piece, mut transform) in &mut pieces {
                *transform = place(&frame.bodies[piece.0]);
            }
        }
    }
    // Gizmos last a frame: marks are drawn every redraw
    if let Some(frame) = last.as_ref() {
        for mark in &frame.marks {
            match *mark {
                Mark::Circle { at, radius, color } => {
                    gizmos.circle_2d(Isometry2d::from_translation(at), radius, color);
                }
                Mark::Line { from, to, color } => gizmos.line_2d(from, to, color),
            }
        }
    }
}

fn place(body: &Body) -> Transform {
    let z = if body.dynamic { 1.0 } else { 0.0 };
    Transform::from_xyz(body.at.x, body.at.y, z).with_rotation(Quat::from_rotation_z(body.angle))
}

/// Point the camera at everything in the frame, with a margin
fn fit(frame: &Frame, camera: &mut CameraQuery) {
    let mut points: Vec<Vec2> = Vec::new();
    for body in &frame.bodies {
        let reach = match body.outline {
            Outline::Circle(r) => Vec2::splat(r),
            Outline::Rect(half) => Vec2::splat(half.length()),
        };
        points.extend([body.at - reach, body.at + reach]);
    }
    for mark in &frame.marks {
        match *mark {
            Mark::Circle { at, radius, .. } => points.extend([at - radius, at + radius]),
            Mark::Line { from, to, .. } => points.extend([from, to]),
        }
    }
    let Some(&first) = points.first() else { return };
    let (lo, hi) = points.iter().fold((first, first), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    let (center, size) = ((lo + hi) / 2.0, (hi - lo) * 1.1 + 1.0);
    if let Ok((mut transform, mut projection)) = camera.single_mut() {
        transform.translation = center.extend(transform.translation.z);
        *projection = Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::AutoMin { min_width: size.x, min_height: size.y },
            ..OrthographicProjection::default_2d()
        });
    }
}

fn end_with_eval(link: Res<Link>, mut exit: MessageWriter<AppExit>) {
    if link.0.eval_done.load(Ordering::Relaxed) {
        exit.write(AppExit::Success);
    }
}
