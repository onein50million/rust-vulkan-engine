use std::{mem::size_of, ptr::addr_of};

use crate::{renderer::{DanielError, drawables::Drawable}, support::PostProcessPushConstants};
use erupt::vk;

use super::VulkanData;

impl VulkanData{
    pub fn draw_frame(&mut self) -> Result<(), DanielError> {
        if !self.swapchain_created {
            match self.recreate_swapchain() {
                Ok(_) => {}
                Err(_) => return Err(DanielError::SwapchainNotCreated),
            }
        }

        let fences = [self.in_flight_fence.unwrap()];
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .wait_for_fences(&fences, true, u64::MAX)
        }
        .unwrap();
        unsafe { self.device.as_ref().unwrap().reset_fences(&fences) }.unwrap();

        self.run_commands();

        match unsafe {
            self.device.as_ref().unwrap().acquire_next_image_khr(
                self.swapchain.unwrap(),
                u64::MAX,
                Some(self.image_available_semaphore.unwrap()),
                None,
            )
        }
        .result()
        {
            Ok(index) => self.image_index = index,
            Err(e) => {
                if e == vk::Result::ERROR_OUT_OF_DATE_KHR || e == vk::Result::SUBOPTIMAL_KHR {
                    unsafe { self.device.as_ref().unwrap().device_wait_idle() }.unwrap();
                    self.cleanup_swapchain();
                    match self.recreate_swapchain() {
                        Ok(_) => return Ok(()),
                        Err(error) => {
                            return Err(error);
                        }
                    }
                } else {
                    panic!("acquire_next_image error");
                }
            }
        };

        let wait_semaphores = [self.image_available_semaphore.unwrap()];
        let signal_semaphores = [self.render_finished_semaphore.unwrap()];
        let wait_stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.command_buffer.unwrap()];
        let submits = [vk::SubmitInfoBuilder::new()
            .command_buffers(command_buffers)
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .wait_dst_stage_mask(&wait_stages)];

        unsafe {
            self.device.as_ref().unwrap().queue_submit(
                self.main_queue.unwrap(),
                &submits,
                Some(self.in_flight_fence.unwrap()),
            )
        }
        .unwrap();

        let swapchains = [self.swapchain.unwrap()];
        let image_indices = [self.image_index];
        let present_info = vk::PresentInfoKHRBuilder::new()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&signal_semaphores);

        match unsafe {
            self.device
                .as_ref()
                .unwrap()
                .queue_present_khr(self.main_queue.unwrap(), &present_info)
        }
        .result()
        {
            Ok(_) => {}
            Err(e) => {
                if e == vk::Result::ERROR_OUT_OF_DATE_KHR || e == vk::Result::SUBOPTIMAL_KHR {
                    unsafe { self.device.as_ref().unwrap().device_wait_idle() }.unwrap();
                    self.cleanup_swapchain();
                    self.recreate_swapchain()?;
                }
            }
        }

        Ok(())
    }
    fn run_commands(&self) {
                let command_buffer_begin_info = vk::CommandBufferBeginInfoBuilder::new();
                let command_buffer = self.command_buffer.as_ref().unwrap();
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .begin_command_buffer(*command_buffer, &command_buffer_begin_info)
                }
                .unwrap();

                let clear_colors = vec![
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    },
                ];

                let render_pass_info = vk::RenderPassBeginInfoBuilder::new()
                    .render_pass(self.render_pass.unwrap())
                    .framebuffer(self.framebuffers[self.image_index as usize])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.surface_capabilities.unwrap().current_extent,
                    })
                    .clear_values(&clear_colors);

                unsafe {
                    self.device.as_ref().unwrap().cmd_begin_render_pass(
                        *command_buffer,
                        &render_pass_info,
                        vk::SubpassContents::INLINE,
                    )
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.graphics_pipelines[0],
                    )
                };

                let vertex_buffers = [self.vertex_buffer.unwrap()];
                let offsets = [0 as vk::DeviceSize];
                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_vertex_buffers(
                        *command_buffer,
                        0,
                        &vertex_buffers,
                        &offsets,
                    )
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_index_buffer(
                        *command_buffer,
                        self.index_buffer.unwrap(),
                        0 as vk::DeviceSize,
                        vk::IndexType::UINT32,
                    )
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout.unwrap(),
                        0,
                        self.descriptor_sets.as_ref().unwrap(),
                        &[],
                    )
                };

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_set_depth_test_enable_ext(*command_buffer, false)
                };

                self.cubemap.as_ref().unwrap().draw(
                    self.device.as_ref().unwrap(),
                    *command_buffer,
                    self.pipeline_layout.unwrap(),
                );

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_set_depth_test_enable_ext(*command_buffer, true)
                };

                for object in &self.objects {
                    object.draw(
                        self.device.as_ref().unwrap(),
                        *command_buffer,
                        self.pipeline_layout.unwrap(),
                    );
                }

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_set_depth_test_enable_ext(*command_buffer, false)
                };

                // for object in &self.fullscreen_quads {
                //     object.draw(
                //         self.device.as_ref().unwrap(),
                //         *command_buffer,
                //         self.pipeline_layout.unwrap(),
                //     );
                // }

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_set_depth_test_enable_ext(*command_buffer, true)
                };
                
                unsafe{
                    self.device.as_ref().unwrap().cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.graphics_pipelines[1],
                    );
                    self.device.as_ref().unwrap().cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.postprocess_subpass_pipeline_layout.unwrap(),
                        0,
                        &self.postprocess_descriptor_sets.as_ref().unwrap(),
                        &[],
                    );
                    self.device.as_ref().unwrap().cmd_next_subpass(*command_buffer, vk::SubpassContents::INLINE);

                    let push_constants = PostProcessPushConstants{
                        view_inverse: self.uniform_buffer_object.view.try_inverse().unwrap(),
                        time: self.uniform_buffer_object.time,
                    };

                    self.device.as_ref().unwrap().cmd_push_constants(*command_buffer, self.postprocess_subpass_pipeline_layout.unwrap(), vk::ShaderStageFlags::FRAGMENT, 0, size_of::<PostProcessPushConstants>() as u32, addr_of!(push_constants) as _);
                    self.device.as_ref().unwrap().cmd_draw(*command_buffer, 3, 1, 0, 0);
                }

                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.ui_data.pipeline.unwrap(),
                    )
                };

                let vertex_buffers = [self.ui_data.vertex_buffer.unwrap()];
                let offsets = [0 as vk::DeviceSize];
                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_vertex_buffers(
                        *command_buffer,
                        0,
                        &vertex_buffers,
                        &offsets,
                    )
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_index_buffer(
                        *command_buffer,
                        self.ui_data.index_buffer.unwrap(),
                        0 as vk::DeviceSize,
                        vk::IndexType::UINT32,
                    )
                };

                let descriptor_sets = [self.ui_data.descriptor_set.unwrap()];
                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.ui_data.pipeline_layout.unwrap(),
                        0,
                        &descriptor_sets,
                        &[],
                    )
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_draw_indexed(
                        *command_buffer,
                        self.ui_data.num_indices,
                        1,
                        0,
                        0,
                        0,
                    )
                };


                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_end_render_pass(*command_buffer)
                };

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .end_command_buffer(*command_buffer)
                }
                .unwrap();
            
    }

}