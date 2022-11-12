use nalgebra::{Translation3, UnitQuaternion, Scale3, Matrix4};

use crate::support::map_range_linear;



pub(crate) struct AnimationObject {
    pub(crate) frame_start: usize,
    pub(crate) frame_count: usize,
}

#[derive(Copy, Clone)]
pub struct Keyframe {
    pub frame_time: f32,
    pub translation: Option<Translation3<f32>>,
    pub rotation: Option<UnitQuaternion<f32>>,
    pub scale: Option<Scale3<f32>>,
}
impl Keyframe {
    fn to_homogeneous(&self) -> Matrix4<f32> {
        let translation = self.translation.unwrap_or_default();
        let rotation = self.rotation.unwrap_or_default();
        let scale = self.scale.unwrap_or(Scale3::identity());
        translation.to_homogeneous() * rotation.to_homogeneous() * scale.to_homogeneous()
    }
}

#[derive(Clone)]
pub struct AnimationKeyframes {
    pub keyframes: Vec<Keyframe>,
    pub end_time: f32,
}

trait ClampedAddition {
    fn clamped_addition(self, amount: i64, min: Self, max: Self) -> Self;
}

impl ClampedAddition for usize {
    fn clamped_addition(self, amount: i64, min: Self, max: Self) -> Self {
        let new_value = self as i64 + amount;

        if new_value < Self::MIN as i64 {
            Self::MIN
        } else if new_value > Self::MAX as i64 {
            Self::MAX
        } else {
            new_value as usize
        }
        .clamp(min, max)
    }
}

impl AnimationKeyframes {
    pub fn get_closest_below(&self, target_frame_time: f32) -> Option<usize> {
        let output = ((self.keyframes.len() as f32) * (target_frame_time / self.end_time)) as usize;
        if output < self.keyframes.len() {
            Some(output)
        } else {
            None
        }
    }

    pub fn get_closest_above(&self, target_frame_time: f32) -> Option<usize> {
        let output =
            (((self.keyframes.len() as f32) * (target_frame_time / self.end_time)).ceil()) as usize;
        if output < self.keyframes.len() {
            Some(output)
        } else {
            None
        }
    }

    pub fn sample(&self, index: f32) -> Matrix4<f32> {
        let below = self.get_closest_below(index);
        let above = self.get_closest_above(index);
        match (below, above) {
            (None, None) => Matrix4::identity(),
            (Some(below), None) => self.keyframes[below].to_homogeneous(),
            // (None, Some(above)) => { self.keyframes[above].to_homogeneous() }
            (None, Some(above)) => self.keyframes[above].to_homogeneous(),
            (Some(below), Some(above)) => {
                self.interpolate(above, below, index).to_homogeneous()
                // self.keyframes[below].to_homogeneous()
            }
        }
    }
    pub fn interpolate(&self, first: usize, second: usize, frame_time: f32) -> Keyframe {
        let first = self.keyframes[first];
        let second = self.keyframes[second];

        let mapped_range =
            map_range_linear(frame_time, first.frame_time, second.frame_time, 0.0, 1.0);
        let mapped_range = match mapped_range.is_nan() {
            true => 0.5,
            false => mapped_range,
        };

        let translation = Some(match (first.translation, second.translation) {
            (None, None) => Translation3::identity(),
            (Some(first), None) => first,
            (None, Some(second)) => second,
            (Some(first), Some(second)) => Translation3::from(
                first.vector * (1.0 - mapped_range) + second.vector * mapped_range,
            ),
        });
        let rotation = Some(match (first.rotation, second.rotation) {
            (None, None) => UnitQuaternion::identity(),
            (Some(first), None) => first,
            (None, Some(second)) => second,
            (Some(first), Some(second)) => first.slerp(&second, mapped_range),
        });
        let scale = Some(match (first.scale, second.scale) {
            (None, None) => Scale3::identity(),
            (Some(first), None) => first,
            (None, Some(second)) => second,
            (Some(first), Some(second)) => {
                Scale3::from(first.vector * (1.0 - mapped_range) + second.vector * mapped_range)
            }
        });

        Keyframe {
            frame_time,
            translation,
            rotation,
            scale,
        }
    }

    pub fn frametime(&self, index: usize) -> f32 {
        if index < self.keyframes.len() {
            self.keyframes[index].frame_time
        } else {
            f32::NAN
        }
    }
    pub fn add_sample(&mut self, sample: Keyframe) {
        match self.keyframes.binary_search_by(|keyframe| {
            keyframe.frame_time.partial_cmp(&sample.frame_time).unwrap()
        }) {
            Ok(index) => {
                match (self.keyframes[index].translation, sample.translation) {
                    (_, None) => {}
                    (None, Some(translation)) => {
                        self.keyframes[index].translation = Some(translation)
                    }
                    (Some(_current_translation), Some(_new_translation)) => { /*TODO, maybe do some fancy averaging but shouldn't happen too often so we should be good to ignore it*/
                    }
                };
                match (self.keyframes[index].rotation, sample.rotation) {
                    (_, None) => {}
                    (None, Some(rotation)) => self.keyframes[index].rotation = Some(rotation),
                    (Some(_), Some(_)) => { /*TODO*/ }
                };
                match (self.keyframes[index].scale, sample.scale) {
                    (_, None) => {}
                    (None, Some(scale)) => self.keyframes[index].scale = Some(scale),
                    (Some(_), Some(_)) => { /*TODO*/ }
                }
            }
            Err(index) => {
                self.keyframes.insert(index, sample);
            }
        }
    }
}
