use std::{marker::PhantomData, mem::size_of, ffi::c_void};

use erupt::vk;

use crate::support::{Vertex, UniformBufferObject, ShaderStorageBufferObject};

use super::vulkan_data::VulkanData;

pub struct MappedBufferIterator<T> {
    ptr: *const T,
    end: *const T,
}

impl<T> Iterator for MappedBufferIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            let old = self.ptr;
            self.ptr = unsafe { self.ptr.offset(1) };

            Some(unsafe { std::ptr::read(old) })
        }
    }
}

pub struct MappedBufferIteratorRef<'a, T> {
    ptr: *const T,
    end: *const T,
    phantom: PhantomData<&'a T>,
}

impl<'a, T> Iterator for MappedBufferIteratorRef<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            let old = self.ptr;
            self.ptr = unsafe { self.ptr.offset(1) };

            Some(unsafe { &*old })
        }
    }
}

pub struct MappedBufferIteratorMutRef<'a, T> {
    ptr: *mut T,
    end: *mut T,
    phantom: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for MappedBufferIteratorMutRef<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            let old = self.ptr;
            self.ptr = unsafe { self.ptr.offset(1) };

            Some(unsafe { &mut *old })
        }
    }
}

pub struct UnmappedBuffer<T> {
    pub buffer: vk::Buffer,
    count: usize,
    allocation: vk_mem_erupt::Allocation,
    allocation_info: vk_mem_erupt::AllocationInfo,
    phantom: PhantomData<T>,
}
impl<T> UnmappedBuffer<T> {
    pub fn new(vulkan_data: &VulkanData, usage_flags: vk::BufferUsageFlags, buffer_data: &[T]) -> Self {
        let (transfer_buffer, transfer_allocation, transfer_allocation_info) = {
            let buffer_info = vk::BufferCreateInfoBuilder::new()
                .size((buffer_data.len() * size_of::<T>()) as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);
            let allocation_info = vk_mem_erupt::AllocationCreateInfo {
                usage: vk_mem_erupt::MemoryUsage::CpuToGpu,
                flags: vk_mem_erupt::AllocationCreateFlags::MAPPED,
                required_flags: vk::MemoryPropertyFlags::empty(),
                preferred_flags: vk::MemoryPropertyFlags::empty(),
                memory_type_bits: u32::MAX,
                pool: None,
                user_data: None,
            };

            vulkan_data
                .allocator
                .as_ref()
                .unwrap()
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Transfer buffer failed")
        };

        unsafe {
            (transfer_allocation_info.get_mapped_data() as *mut T)
                .copy_from_nonoverlapping(buffer_data.as_ptr(), buffer_data.len());
        }

        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size((buffer_data.len() * size_of::<T>()) as u64)
            .usage(usage_flags | vk::BufferUsageFlags::TRANSFER_DST);
        let allocation_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::GpuOnly,
            flags: vk_mem_erupt::AllocationCreateFlags::empty(),
            required_flags: vk::MemoryPropertyFlags::empty(),
            preferred_flags: vk::MemoryPropertyFlags::empty(),
            memory_type_bits: u32::MAX,
            pool: None,
            user_data: None,
        };

        let (buffer, allocation, allocation_info) = vulkan_data
            .allocator
            .as_ref()
            .unwrap()
            .create_buffer(&buffer_info, &allocation_info)
            .expect("Unmapped buffer creation failed");

        vulkan_data.copy_buffer(
            transfer_buffer,
            buffer,
            (buffer_data.len() * size_of::<T>()) as u64,
        );

        vulkan_data
            .allocator
            .as_ref()
            .unwrap()
            .destroy_buffer(transfer_buffer, &transfer_allocation);

        Self {
            buffer,
            count: buffer_data.len(),
            allocation,
            allocation_info,
            phantom: PhantomData,
        }
    }
    pub fn destroy(&mut self, vulkan_data: &VulkanData) {
        vulkan_data
            .allocator
            .as_ref()
            .unwrap()
            .destroy_buffer(self.buffer, &self.allocation);
    }
}


pub struct MappedBuffer<T> {
    pub(crate) buffer: vk::Buffer,
    pub(crate) allocation: vk_mem_erupt::Allocation,
    pub(crate) allocation_info: vk_mem_erupt::AllocationInfo,
    pub(crate) phantom: PhantomData<T>,
}
impl<T> MappedBuffer<T>{
    pub fn new(vulkan_data: &VulkanData, usage: vk::BufferUsageFlags) -> Self{
        let buffer_info = vk::BufferCreateInfoBuilder::new()
        .size(size_of::<T>() as u64)
        .usage(usage);
    let allocation_info = vk_mem_erupt::AllocationCreateInfo {
        usage: vk_mem_erupt::MemoryUsage::CpuToGpu,
        flags: vk_mem_erupt::AllocationCreateFlags::MAPPED,
        required_flags: vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE,
        preferred_flags: vk::MemoryPropertyFlags::empty(),
        memory_type_bits: u32::MAX,
        pool: None,
        user_data: None,
    };

    let (buffer, allocation, allocation_info) = vulkan_data
        .allocator
        .as_ref()
        .unwrap()
        .create_buffer(&buffer_info, &allocation_info)
        .expect("MappedBuffer failed to allocate");
    let mut out_buffer = MappedBuffer {
        buffer,
        allocation,
        allocation_info,
        phantom: PhantomData,
    };
    out_buffer
    }
    pub fn get_mut(&mut self) -> &mut T{
        unsafe{
            &mut *(self.allocation_info.get_mapped_data() as *mut T)
        }
    }
}


impl VulkanData{
    pub(crate) fn create_index_buffer(&mut self) {
        let buffer_size: vk::DeviceSize = (std::mem::size_of::<u32>() * self.indices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer_with_memory(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let mut destination_pointer = std::ptr::null::<*mut c_void>() as *mut c_void;
        unsafe {
            self.device.as_ref().unwrap().map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                None,
                &mut destination_pointer as *mut *mut c_void,
            )
        }
        .unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.indices.as_ptr() as *mut c_void,
                destination_pointer,
                buffer_size as usize,
            )
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory)
        };

        let (index_buffer, index_buffer_memory) = self.create_buffer_with_memory(
            buffer_size + 40,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.index_buffer = Some(index_buffer);
        self.index_buffer_memory = Some(index_buffer_memory);
        self.copy_buffer(staging_buffer, self.index_buffer.unwrap(), buffer_size);

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(Some(staging_buffer), None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .free_memory(Some(staging_buffer_memory), None)
        };
    }
    pub(crate) fn create_vertex_buffer(&mut self) {
        let buffer_size: vk::DeviceSize =
            (std::mem::size_of::<Vertex>() * self.vertices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer_with_memory(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let mut destination_pointer = std::ptr::null::<*mut c_void>() as *mut c_void;
        unsafe {
            self.device.as_ref().unwrap().map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                None,
                &mut destination_pointer as *mut *mut c_void,
            )
        }
        .unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.vertices.as_ptr() as *mut c_void,
                destination_pointer,
                buffer_size as usize,
            )
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory)
        };

        let (vertex_buffer, vertex_buffer_memory) = self.create_buffer_with_memory(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.vertex_buffer = Some(vertex_buffer);
        self.vertex_buffer_memory = Some(vertex_buffer_memory);
        self.copy_buffer(staging_buffer, self.vertex_buffer.unwrap(), buffer_size);

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(Some(staging_buffer), None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .free_memory(Some(staging_buffer_memory), None)
        };
    }
    pub fn update_vertex_and_index_buffers(&mut self) {
        if self.vertex_buffer.is_some() || self.vertex_buffer_memory.is_some() {
            self.destroy_vertex_buffer();
            self.create_vertex_buffer();
        }
        if self.index_buffer.is_some() || self.index_buffer_memory.is_some() {
            self.destroy_index_buffer();
            self.create_index_buffer();
        }
    }
    #[deprecated(note="Wasteful and not super important")]
    pub(crate) fn copy_buffer(
        &self,
        source_buffer: vk::Buffer,
        destination_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        let command_buffer = self.begin_single_time_commands();
        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer(
                command_buffer,
                source_buffer,
                destination_buffer,
                &[vk::BufferCopyBuilder::new().size(size)],
            )
        };
        self.end_single_time_commands(command_buffer);
    }

    #[deprecated(note="Use VMA instead")]
    pub(crate) fn create_buffer_with_memory(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_buffer(&buffer_info, None)
        }
        .unwrap();

        let memory_requirements = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_buffer_memory_requirements(buffer)
        };

        let allocate_info = vk::MemoryAllocateInfoBuilder::new()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                self.find_memory_type(memory_requirements.memory_type_bits, properties),
            );

        let buffer_memory = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_memory(&allocate_info, None)
        }
        .unwrap();

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .bind_buffer_memory(buffer, buffer_memory, 0)
        }
        .unwrap();

        return (buffer, buffer_memory);
    }
    pub(crate) fn create_buffers(&mut self) {
        let device_size = (std::mem::size_of::<UniformBufferObject>()) as vk::DeviceSize;

        let buffer_create_info = vk::BufferCreateInfoBuilder::new()
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .size(device_size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_create_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::CpuToGpu,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };

        let (buffer, allocation, _) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .unwrap();
        self.uniform_buffers.push(buffer);
        self.uniform_buffer_allocations.push(allocation);
        self.uniform_buffer_pointers.push(
            self.allocator
                .as_ref()
                .unwrap()
                .map_memory(&allocation)
                .unwrap(),
        );

        self.post_process_ubo = Some(MappedBuffer::new(&self, vk::BufferUsageFlags::UNIFORM_BUFFER));

        let device_size = (std::mem::size_of::<ShaderStorageBufferObject>()) as vk::DeviceSize;

        let buffer_create_info = vk::BufferCreateInfoBuilder::new()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .size(device_size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_create_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::GpuOnly,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };

        let (buffer, allocation, _) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .unwrap();
        self.storage_buffer = Some(buffer);
        self.storage_buffer_allocation = Some(allocation);
    }

    pub(crate) fn transfer_data_to_storage_buffer(&self, data: &ShaderStorageBufferObject) {
        let (staging_buffer, _allocation, allocation_info) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_buffer(
                &vk::BufferCreateInfoBuilder::new()
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .size(size_of::<ShaderStorageBufferObject>() as vk::DeviceSize)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem_erupt::AllocationCreateInfo {
                    usage: vk_mem_erupt::MemoryUsage::CpuOnly,
                    flags: vk_mem_erupt::AllocationCreateFlags::MAPPED,
                    ..Default::default()
                },
            )
            .unwrap();

        unsafe {
            allocation_info.get_mapped_data().copy_from_nonoverlapping(
                data as *const ShaderStorageBufferObject as _,
                size_of::<ShaderStorageBufferObject>(),
            )
        }
        let command_buffer = self.begin_single_time_commands();

        let regions = vk::BufferCopyBuilder::new()
            .size(size_of::<ShaderStorageBufferObject>() as vk::DeviceSize)
            .dst_offset(0)
            .src_offset(0);

        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer(
                command_buffer,
                staging_buffer,
                self.storage_buffer.unwrap(),
                &[regions],
            )
        }

        self.end_single_time_commands(command_buffer)
    }
}