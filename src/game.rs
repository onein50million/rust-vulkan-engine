use cgmath::{Vector3, Quaternion, Matrix4, Transform, One};
use crate::RenderObject;


pub(crate) struct GameObject{
    position:Vector3<f32>,
    rotation:Quaternion<f32>,
    render_object: RenderObject,
}
impl GameObject{
    fn process(&mut self, delta_time: f32){
        //Do stuff
    }
}

pub(crate) struct Camera{
    position:Vector3<f32>,
    rotation:Quaternion<f32>
}

impl Camera{
    pub(crate) fn get_view_matrix_no_translation(&self) -> Matrix4<f32>{
        let matrix = Matrix4::from(self.rotation);
        return matrix.inverse_transform().unwrap();
    }

    pub(crate) fn get_view_matrix(&self) -> Matrix4<f32>{
        let matrix = Matrix4::from_translation(self.rotation*self.position);
        return matrix.inverse_transform().unwrap();
    }

}

pub(crate) struct Game{
    camera: Camera,
    objects: Vec<GameObject>
}

impl Game{
    fn new() -> Self{
        return Game{
            camera: Camera{
                position: Vector3::new(0.0,0.0,0.0),
                rotation: Quaternion::one(),
            },
            objects: vec![]
        }
    }
    fn process(&mut self, delta_time: f32){
        for i in 0..self.objects.len(){
            self.objects[i].process(delta_time);
        }
    }
}