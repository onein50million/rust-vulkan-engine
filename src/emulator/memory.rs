use std::borrow::Borrow;
use std::convert::TryInto;
use crate::renderer::VulkanData;

#[derive(Copy,Clone, Debug)]
enum SegmentTypes {
    Ram,
    ConnectedDevices,
    Device,
    Unknown,
}
#[derive(Debug)]
struct Segment{
    segment_type: SegmentTypes,
    size: usize,
}

const MAX_DEVICES: usize = 64;
const MAX_DEVICE_SIZE: usize = 1024;

const SEGMENTS: [Segment; 3] = [
    Segment{segment_type: SegmentTypes::Ram, size: 0xFFFF },
    Segment{segment_type: SegmentTypes::ConnectedDevices, size: MAX_DEVICES*8},
    Segment{segment_type: SegmentTypes::Device, size: MAX_DEVICES*MAX_DEVICE_SIZE},
];

fn return_segment(address: usize) -> SegmentTypes{
    let mut i = 0;
    while i < SEGMENTS.len(){
        // println!("Start: {:}", SEGMENT_STARTS[i]);
        // println!("Size: {:}", SEGMENTS[i].size);
        // println!("address: {:}", address);
        if address >= SEGMENT_STARTS[i] && address < SEGMENT_STARTS[i] + SEGMENTS[i].size{
            return SEGMENTS[i].segment_type;
        }
        i += 1;
    }
    return SegmentTypes::Unknown;

}


const fn calculate_start(index: usize) -> usize{
    let mut sum = 0;
    let mut i = 0;

    while i<index{
        sum += SEGMENTS[i].size;
        i+=1;
    }
    return sum;
}

const SEGMENT_STARTS: [usize; SEGMENTS.len()] = [
    calculate_start(0),
    calculate_start(1),
    calculate_start(2),
];

pub(crate) struct Display{
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) cpu_image_index: usize
}
impl Display{
    const TYPE: u8 = 1;

    fn get_type(&self) -> u8 {
        return Self::TYPE
    }

    fn get_bytes(&self, address: usize, num_bytes: usize, vulkan_data: &crate::renderer::VulkanData) -> Vec<u8> {
        let data = vulkan_data.cpu_images[self.cpu_image_index].get_data();
        return data[address..][..num_bytes].to_owned();
    }

    fn set_bytes(&mut self, address: usize, num_bytes: usize, bytes: Vec<u8>, vulkan_data: &mut crate::renderer::VulkanData) {

        let mut data = vulkan_data.cpu_images[self.cpu_image_index].get_data();
        data.splice(address..(address + num_bytes), bytes);
        vulkan_data.cpu_images[self.cpu_image_index].write_data(data)

    }
}

pub(crate) enum Device{
    PlaceHolder,
    Display(Display)
}

#[derive(Copy, Clone)]
struct PlaceholderDevice{
}
impl PlaceholderDevice{
    fn get_type(&self) -> u64 {
        return 0;
    }

    fn get_bytes(&self, address: usize, num_bytes: usize) -> Vec<u8> {
        println!("Get bytes called on placeholder device");
        return std::iter::repeat(69).take(num_bytes).collect();
    }

    fn set_bytes(&mut self, _address: usize, _num_bytes: usize, _bytes: Vec<u8>) {
        println!("Set bytes called on placeholder device")
    }
}

pub struct Memory {
    pub ram: Vec<u8>,
    devices: Vec<Device>,
}

impl Memory{
    pub fn new(initial_bytes: Vec<u8>)-> Self{

        const RAM_INDEX: usize = 0; //TODO: make this smarter

        assert!(matches!(SEGMENTS[RAM_INDEX].segment_type,SegmentTypes::Ram));
        let mut ram = vec![0; SEGMENTS[RAM_INDEX].size];
        ram.splice(0..initial_bytes.len(), initial_bytes);

        let mut devices = Vec::with_capacity(MAX_DEVICES);

        for _i in 0..MAX_DEVICES{
            devices.push(
                Device::PlaceHolder
            )
        }
        let memory = Self{
            ram,
            devices
        };
        return memory;
    }

    pub fn get_ram_length(&self) -> usize{
        return self.ram.len();
    }

    pub(crate) fn get_bytes(&self, address: usize, num_bytes: usize, vulkan_data: &crate::renderer::VulkanData) -> Vec<u8> {
        // println!("segment: {:?}", return_segment(address));
        // println!("address: {:}", address);
        match return_segment(address){
            SegmentTypes::Ram => {
                return self.ram[address..(address+num_bytes).min(SEGMENT_STARTS[1])].to_vec();
            }
            SegmentTypes::ConnectedDevices => {
                let local_address = address - SEGMENT_STARTS[1];
                match &self.devices[local_address/MAX_DEVICE_SIZE]{
                    Device::PlaceHolder => {
                        return std::iter::repeat(0).take(num_bytes).collect()
                    }
                    Device::Display(display) => {
                        return std::iter::repeat(display.get_type()).take(num_bytes).collect()
                    }
                }
            }
            SegmentTypes::Device => {
                let local_address = address - SEGMENT_STARTS[2];
                match &self.devices[local_address/MAX_DEVICE_SIZE]{
                    Device::PlaceHolder => {
                        return std::iter::repeat(0x69).take(num_bytes).collect()
                    }
                    Device::Display(display) => {
                        println!("Display!");
                        return display.get_bytes(local_address%MAX_DEVICE_SIZE, num_bytes, vulkan_data)
                    }
                }
            }
            SegmentTypes::Unknown => {
                println!("warning: unknown memory");
                return 0xCafeDeadBeefu64.to_le_bytes()[..num_bytes].to_vec();
            }
        }
    }

    pub(crate) fn set_bytes(&mut self, address: usize, num_bytes: usize, bytes: Vec<u8>, vulkan_data: &mut crate::renderer::VulkanData) {
        match return_segment(address) {
            SegmentTypes::Ram => {
                for i in 0..num_bytes {
                    self.ram[address + i] = bytes[i];
                }
            }
            SegmentTypes::ConnectedDevices => {}
            SegmentTypes::Device => {
                let local_address = address - SEGMENT_STARTS[2];
                match &mut self.devices[local_address/MAX_DEVICE_SIZE]{
                    Device::PlaceHolder => {}
                    Device::Display(display) => {
                        display.set_bytes(local_address%MAX_DEVICE_SIZE, num_bytes, bytes, vulkan_data)
                    }
                }
            }
            SegmentTypes::Unknown => {
                println!("warning: unknown memory")
            }
        }
    }

    pub(crate) fn set_device(&mut self, index: usize, new_device: Device){
        self.devices[index] = new_device;
    }
    pub(crate) fn get_device(&self, index: usize) -> &Device{
        return &self.devices[index];
    }


}