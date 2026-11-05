//! Predict the moving target and return an aiming command in [`aim`].
//! World axes: +X forward, +Y left, +Z up. Distances are metres and angles are radians.
//! The simulator calls `aim` every 10 ms with a perfect observation captured in the past.

pub mod sim;
pub mod ui;

/// Fixed launcher physics and timing shared by the controller, viewer, and evaluator.
pub mod physics {
    /// Stationary muzzle position in world metres.
    pub const MUZZLE_M: [f64; 3] = [0.0, 0.0, 0.3];
    /// Projectile launch speed in metres per second.
    pub const MUZZLE_SPEED_M_S: f64 = 25.0;
    /// Downward acceleration in metres per second squared.
    pub const GRAVITY_M_S2: f64 = 9.81;
    /// Time from observation capture to delivery.
    pub const OBSERVATION_DELAY_S: f64 = 0.05;
    /// Time from command acceptance to projectile launch.
    pub const LAUNCH_DELAY_S: f64 = 0.02;
    /// Minimum interval between accepted shots.
    pub const FIRE_PERIOD_S: f64 = 0.25;
    /// Projectile lifetime after launch.
    pub const MAX_FLIGHT_S: f64 = 2.0;
    /// Fixed interval between controller calls.
    pub const CONTROL_PERIOD_S: f64 = 0.01;
}

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;
use ui::{Color, Gizmos};

/// Implement latency compensation, plate selection, and future-position aiming.
pub fn aim(input: &Input, debug: &mut Gizmos) -> Option<Command> {
    let resolved = input.stats.hits + input.stats.misses;
    debug.overlay("Hit rate", format!("{}/{resolved}", input.stats.hits));

    let observation = input.observation?;
    for plate in observation.target.plates(input.size) {
        debug.plate(plate, input.size, Color::Cyan);
    }

    let age = input.now_s - observation.time_s;
    let launch_delay = physics::LAUNCH_DELAY_S;
    let latency = age + launch_delay;

    let target = observation.target;
    // 水平距离 / 弹速，先粗估飞行时间
    let dx = target.x_m - physics::MUZZLE_M[0];
    let dy = target.y_m - physics::MUZZLE_M[1];
    let rough_flight_time = dx.hypot(dy) / physics::MUZZLE_SPEED_M_S;
    let predicted_target = target.at(latency + rough_flight_time);
    let plates = predicted_target.plates(input.size);
    let mut selected = None;
    let mut best_score = 0.0;
    for (index, plate) in plates.iter().enumerate() {
        // 从装甲板中心指向炮口的向量
        let to_muzzle: [f64; 3] = std::array::from_fn(|i| physics::MUZZLE_M[i] - plate.center_m[i]);
        let distance = to_muzzle.iter().map(|v| v * v).sum::<f64>().sqrt();
        if distance <= 0.0 {
            continue;
        }
        // 两个方向越一致，评分越大，表示板面越正对炮口
        let score = (0..3)
            .map(|i| plate.outward_normal[i] * to_muzzle[i] / distance)
            .sum::<f64>();

        if score > best_score {
            best_score = score;
            selected = Some((index, *plate));
        }
    }
    let (selected_index, selected_plate) = selected?;

    let mut trajectory = Trajectory::solve(selected_plate.center_m)?;
    let mut flight_time = trajectory.flight_time_s;
    let mut converged = false;

    for _ in 0..10 {
        // 用当前飞行时间，重新预测同一块板
        let target = observation.target.at(latency + flight_time);
        let plate = target.plates(input.size)[selected_index];

        // 重新求弹道
        let new_trajectory = Trajectory::solve(plate.center_m)?;
        let new_time = new_trajectory.flight_time_s;

        let difference = (new_time - flight_time).abs();
        trajectory = new_trajectory;
        flight_time = new_time;

        if difference < 0.0001 {
            converged = true;
            break;
        }
    }

    if !converged {
        return None;
    }

    // 用最终飞行时间重新计算命中时的板姿态和速度。
    let predicted_plate = observation
        .target
        .at(latency + trajectory.flight_time_s)
        .plates(input.size)[selected_index];
    let projectile_velocity = trajectory.velocity(trajectory.flight_time_s);
    let relative_velocity: [f64; 3] =
        std::array::from_fn(|i| projectile_velocity[i] - predicted_plate.velocity_m_s[i]);
    let relative_speed = relative_velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !relative_speed.is_finite() || relative_speed <= 0.0 {
        return None;
    }

    // 外法线朝外，弹丸从正面射入时，相对速度与法线反向，所以取负号。
    let incidence_score = -(0..3)
        .map(|i| predicted_plate.outward_normal[i] * relative_velocity[i])
        .sum::<f64>()
        / relative_speed;
    // 0.5 对应相对入射方向与向内法线夹角不超过 60°，可按效果调整。
    const MIN_INCIDENCE_SCORE: f64 = 0.5;
    let fire = incidence_score.is_finite() && incidence_score >= MIN_INCIDENCE_SCORE;

    // 每一帧画观测板（青色）、预测板（可开火绿色，否则红色）和弹道。
    let plate_color = if fire { Color::Green } else { Color::Red };
    let impact_m = trajectory.position(trajectory.flight_time_s);
    debug.plate(predicted_plate, input.size, plate_color);
    debug.trajectory(trajectory, Color::Orange);
    debug.point("Impact", impact_m, Color::Orange);
    debug.point("Predicted plate", predicted_plate.center_m, plate_color);
    // 紫色线段表示弹丸相对装甲板的入射方向，终点为预计弹道终点。
    let approach_m =
        std::array::from_fn(|i| impact_m[i] - 0.4 * relative_velocity[i] / relative_speed);
    debug.line(approach_m, impact_m, Color::Purple);
    debug.overlay("Plate index", selected_index);
    debug.overlay("Observation age", format!("{:.1} ms", age * 1000.0));
    debug.overlay(
        "Flight",
        format!("{:.1} ms", trajectory.flight_time_s * 1000.0),
    );
    debug.overlay(
        "Incidence score",
        format!("{incidence_score:.3} / min {MIN_INCIDENCE_SCORE:.2}"),
    );
    debug.overlay("Fire", if fire { "Yes" } else { "Hold: incidence angle" });

    Some(Command {
        yaw_rad: trajectory.yaw_rad,
        pitch_rad: trajectory.pitch_rad,
        fire,
    })
}

/// The only simulation data provided to the applicant's controller.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Input {
    pub now_s: f64,
    /// None until the first observation's configured delivery delay has elapsed.
    pub observation: Option<Sample>,
    pub size: Size,
    /// Outcomes resolved by this tick, before the current command is accepted.
    #[serde(default)]
    pub stats: Stats,
}

/// World yaw, upward pitch, and a request to fire at the fixed maximum rate.
/// Angles are latched when a shot is accepted, then applied after `physics::LAUNCH_DELAY_S`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Command {
    pub yaw_rad: f64,
    pub pitch_rad: f64,
    pub fire: bool,
}

/// A gravity-only projectile path. Flight time starts at launch, excluding all delays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trajectory {
    pub yaw_rad: f64,
    pub pitch_rad: f64,
    pub flight_time_s: f64,
}

impl Trajectory {
    /// Solve the low arc to a fixed world point at the fixed muzzle speed.
    /// Returns `None` for an unreachable point, nonfinite input, zero flight time, or timeout.
    /// Predict the moving target separately; this method does not compensate for delays.
    pub fn solve(point_m: [f64; 3]) -> Option<Self> {
        let delta: [f64; 3] = std::array::from_fn(|i| point_m[i] - physics::MUZZLE_M[i]);
        let distance_squared = delta.iter().map(|v| v * v).sum::<f64>();
        let gravity = physics::GRAVITY_M_S2;
        let a = physics::MUZZLE_SPEED_M_S.powi(2) - gravity * delta[2];
        let discriminant = a * a - gravity * gravity * distance_squared;
        if a <= 0.0 || discriminant < 0.0 {
            return None;
        }
        // Rationalized smaller root of the ballistic equation in flight time squared.
        let flight_time_s = (2.0 * distance_squared / (a + discriminant.sqrt())).sqrt();
        if !flight_time_s.is_finite()
            || flight_time_s <= 0.0
            || flight_time_s > physics::MAX_FLIGHT_S
        {
            return None;
        }
        let mut velocity = delta.map(|d| d / flight_time_s);
        velocity[2] += 0.5 * gravity * flight_time_s;
        Some(Self {
            yaw_rad: velocity[1].atan2(velocity[0]),
            pitch_rad: velocity[2].atan2(velocity[0].hypot(velocity[1])),
            flight_time_s,
        })
    }

    /// World position `elapsed_s` after launch; gravity produces a mathematical parabola.
    pub fn position(self, elapsed_s: f64) -> [f64; 3] {
        let velocity = self.velocity(0.0);
        std::array::from_fn(|i| {
            physics::MUZZLE_M[i] + velocity[i] * elapsed_s
                - if i == 2 {
                    0.5 * physics::GRAVITY_M_S2 * elapsed_s * elapsed_s
                } else {
                    0.0
                }
        })
    }

    /// World velocity `elapsed_s` after launch, before subtracting the armor's velocity.
    pub fn velocity(self, elapsed_s: f64) -> [f64; 3] {
        let horizontal = physics::MUZZLE_SPEED_M_S * self.pitch_rad.cos();
        [
            horizontal * self.yaw_rad.cos(),
            horizontal * self.yaw_rad.sin(),
            physics::MUZZLE_SPEED_M_S * self.pitch_rad.sin() - physics::GRAVITY_M_S2 * elapsed_s,
        ]
    }
}

/// World axes are +X forward, +Y left, +Z up. Angles turn from +X toward +Y.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub x_m: f64,
    pub y_m: f64,
    pub theta_rad: f64,
    pub vx_m_s: f64,
    pub vy_m_s: f64,
    pub omega_rad_s: f64,
}

impl Target {
    /// Evaluate constant world velocity and spin without integrating intermediate steps.
    pub fn at(self, elapsed_s: f64) -> Self {
        Self {
            x_m: self.x_m + self.vx_m_s * elapsed_s,
            y_m: self.y_m + self.vy_m_s * elapsed_s,
            theta_rad: (self.theta_rad + self.omega_rad_s * elapsed_s).rem_euclid(TAU),
            ..self
        }
    }

    /// Four equally spaced plates, mounted with their top edges tilted toward the center.
    pub fn plates(self, size: Size) -> [Plate; 4] {
        std::array::from_fn(|slot| {
            let angle = self.theta_rad + slot as f64 * TAU / 4.0;
            let (sin, cos) = angle.sin_cos();
            let (tilt_sin, tilt_cos) = size.mount_pitch_rad.sin_cos();
            Plate {
                center_m: [
                    self.x_m + size.radius_m * cos,
                    self.y_m + size.radius_m * sin,
                    size.plate_center_z_m,
                ],
                outward_normal: [cos * tilt_cos, sin * tilt_cos, tilt_sin],
                right_axis: [-sin, cos, 0.0],
                up_axis: [-cos * tilt_sin, -sin * tilt_sin, tilt_cos],
                velocity_m_s: [
                    self.vx_m_s - self.omega_rad_s * size.radius_m * sin,
                    self.vy_m_s + self.omega_rad_s * size.radius_m * cos,
                    0.0,
                ],
            }
        })
    }
}

/// Scoring rectangle and mounting geometry. Plates have equal radii and heights.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Size {
    /// Positive pitch tilts the outward normal up, matching the main simulator.
    pub mount_pitch_rad: f64,
    pub radius_m: f64,
    pub plate_width_m: f64,
    pub plate_height_m: f64,
    pub plate_center_z_m: f64,
}

/// Ground-truth plate state; plate indices remain attached to the target as it turns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plate {
    pub center_m: [f64; 3],
    pub outward_normal: [f64; 3],
    pub right_axis: [f64; 3],
    pub up_axis: [f64; 3],
    pub velocity_m_s: [f64; 3],
}

impl Plate {
    /// Convert plate coordinates (right, up, outward), in metres, to world coordinates.
    pub fn point(self, right_m: f64, up_m: f64, outward_m: f64) -> [f64; 3] {
        std::array::from_fn(|i| {
            self.center_m[i]
                + self.right_axis[i] * right_m
                + self.up_axis[i] * up_m
                + self.outward_normal[i] * outward_m
        })
    }
}

/// Perfect target state at the original observation capture time.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub time_s: f64,
    pub target: Target,
}

/// Accepted shots and their outcomes. Requests blocked by the firing rate are not shots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Stats {
    pub shots: usize,
    pub hits: usize,
    pub misses: usize,
    pub invalid_commands: usize,
}
