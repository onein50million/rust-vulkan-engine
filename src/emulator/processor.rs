use std::convert::TryInto;
use std::ffi::CString;
use crate::emulator::opcodes;
use crate::emulator::format;
use crate::emulator::memory;
use crate::game::Game;
use crate::renderer::VulkanData;

const DUMP_REGISTERS: bool = false;
pub struct Processor{
    pub registers: [u64; 32],
    pub program_counter: u64,
    pub memory: memory::Memory
}

pub enum ProcessorResult {
    Break,
    Continue
}

impl Processor{
    pub fn new(code: Vec<u8>) -> Self{
        let mut processor = Self{
            registers: [0x0;32],
            program_counter: 0,
            memory: memory::Memory::new(code),
        };
        processor.registers[2] = processor.memory.get_ram_length() as u64; //sp
        processor.registers[8] = processor.registers[2]; //s0
        processor.registers[0] = 0; //x0
        processor.program_counter = 0x0;
        return processor;
    }
    fn dump_registers(&self){
        println!("Registers at {:X?}: {:X?}",self.program_counter, self.registers);
        // let stack_pointer = self.registers[2] as usize;
        // println!("stack_pointer: {:}", stack_pointer);
        // let memory: Vec<_> = self.memory[stack_pointer..][..1024].try_into().unwrap();
        // println!("stack: {:X?}", memory);

    }
    pub(crate) fn get_bytes(&self, address: usize, num_bytes: usize, vulkan_data: &crate::renderer::VulkanData) -> Vec<u8> {
        let mut output = vec![0;num_bytes];
        let new_bytes = self.memory.get_bytes(address, num_bytes, vulkan_data);
        for i in 0..new_bytes.len(){
            output[i] = new_bytes[i];
        }
        return output;
    }
    pub(crate) fn set_bytes(&mut self, address: usize, num_bytes: usize, bytes: Vec<u8>, vulkan_data: &mut crate::renderer::VulkanData){
        self.memory.set_bytes(address, num_bytes, bytes, vulkan_data);
    }

    fn unimplemented_instruction(&self, instruction: u32) {
        let opcode = instruction & 0x7f;
        let r_format = format::R::new(instruction);
        let i_format = format::I::new(instruction);
        let shift_i_format = format::ShiftI::new(instruction);
        let s_format = format::S::new(instruction);
        let b_format = format::B::new(instruction);
        let u_format = format::U::new(instruction);
        let j_format = format::J::new(instruction);


        println!("Unimplemented instruction at {:X}", self.program_counter);
        println!("Registers: {:X?}", self.registers);
        println!("opcode: {:X}", opcode);
        println!("r_format: {:?}", r_format);
        println!("i_format: {:?}", i_format);
        println!("shift_i_format: {:?}", shift_i_format);
        println!("s_format: {:?}", s_format);
        println!("b_format: {:?}", b_format);
        println!("u_format: {:?}", u_format);
        println!("j_format: {:?}", j_format);

    }


    pub(crate) fn process(&mut self, vulkan_data: &mut crate::renderer::VulkanData) -> ProcessorResult {
        let mut program_counter_handled = false;
        if DUMP_REGISTERS{
            self.dump_registers();
        }

        let instruction = self.get_bytes(self.program_counter as usize, 4, vulkan_data).try_into().unwrap();
        let instruction = u32::from_le_bytes(instruction);

        let opcode = instruction & 0x7f;


        match opcode {
            opcodes::LUI => {
                let u_format = format::U::new(instruction);

                let immediate = u64::from_le_bytes(u_format.imm.to_le_bytes());
                self.registers[u_format.rd as usize] = immediate;
            }
            opcodes::AUIPC => {
                let u_format = format::U::new(instruction);

                let offset = u64::from_le_bytes(u_format.imm.to_le_bytes());
                self.registers[u_format.rd as usize] = self.program_counter + offset;

            }
            opcodes::JAL =>{
                //JAL
                // println!("JAL");
                let j_format = format::J::new(instruction);

                self.registers[j_format.rd as usize] = self.program_counter+4;
                self.program_counter = self.program_counter.signed_offset(j_format.imm);
                program_counter_handled = true;
            }
            opcodes::JALR =>{
                //JALR
                // println!("JALR");

                let i_format = format::I::new(instruction);
                self.registers[i_format.rd as usize] = self.program_counter+4;
                let offset = i_format.imm;
                self.program_counter = self.registers[i_format.rs1 as usize].signed_offset(offset) & 0xFF_FF_FF_FF_FF_FF_FF_FE;
                program_counter_handled = true;

            }
            opcodes::BRANCH =>{
                //uses funky b-type encoding
                let b_format = format::B::new(instruction);

                let offset = b_format.imm;
                match b_format.funct3{
                    0b000 => {
                        //BEQ
                        if self.registers[b_format.rs1 as usize] == self.registers[b_format.rs2 as usize] {
                            self.program_counter = self.program_counter.signed_offset(offset);
                            program_counter_handled = true;

                        }
                    }
                    0b001 => {
                        //BNE
                        if self.registers[b_format.rs1 as usize] != self.registers[b_format.rs2 as usize] {
                            self.program_counter = self.program_counter.signed_offset(offset);
                            program_counter_handled = true;

                        }
                    }
                    0b100 => {
                        //BLT
                        if i64::from_le_bytes(self.registers[b_format.rs1 as usize].to_le_bytes()) < i64::from_le_bytes(self.registers[b_format.rs2 as usize].to_le_bytes()) {
                            self.program_counter = self.program_counter.signed_offset(offset);
                            program_counter_handled = true;

                        }
                    }
                    0b101 => {
                        //BGE
                        if i64::from_le_bytes(self.registers[b_format.rs1 as usize].to_le_bytes()) > i64::from_le_bytes(self.registers[b_format.rs2 as usize].to_le_bytes()) {
                            self.program_counter = self.program_counter.signed_offset(offset);
                            program_counter_handled = true;

                        }
                    }
                    0b110 => {
                        //BLTU
                        if self.registers[b_format.rs1 as usize] < self.registers[b_format.rs2 as usize]{
                            self.program_counter = self.program_counter.signed_offset(offset);
                            program_counter_handled = true;

                        }
                    }
                    0b111 => {
                        //BGEU
                        if self.registers[b_format.rs1 as usize] > self.registers[b_format.rs2 as usize] {
                            self.program_counter = self.program_counter.signed_offset(offset);
                            program_counter_handled = true;

                        }
                    }

                    _ => {self.unimplemented_instruction(instruction);}
                }
            }
            opcodes::LOAD => {
                let i_format = format::I::new(instruction);
                let address = self.registers[i_format.rs1 as usize].signed_offset(i_format.imm);
                match i_format.funct3{
                    0b000 =>{
                        //lb
                        let value = i8::from_le_bytes(self.get_bytes(address as usize, 1, vulkan_data).try_into().unwrap());
                        self.registers[i_format.rd as usize] = u64::from_le_bytes((value as i64).to_le_bytes());
                    }
                    0b001 => {
                        //lh
                        let value = i16::from_le_bytes(self.get_bytes(address as usize, 2, vulkan_data).try_into().unwrap());
                        self.registers[i_format.rd as usize] = u64::from_le_bytes((value as i64).to_le_bytes());
                    }
                    0b010 => {
                        //lw
                        let value = i32::from_le_bytes(self.get_bytes(address as usize, 4, vulkan_data).try_into().unwrap());
                        self.registers[i_format.rd as usize] = u64::from_le_bytes((value as i64).to_le_bytes());

                    }
                    0b011 =>{
                        //ld
                        let value = i64::from_le_bytes(self.get_bytes(address as usize, 8, vulkan_data).try_into().unwrap());
                        self.registers[i_format.rd as usize] = u64::from_le_bytes(value.to_le_bytes());
                    }
                    0b100 => {
                        //lbu
                        let value = u8::from_le_bytes(self.get_bytes(address as usize, 1, vulkan_data).try_into().unwrap());
                        self.registers[i_format.rd as usize] = value as u64;
                    }
                    0b101 => {
                        //lhu
                        let value = u16::from_le_bytes(self.get_bytes(address as usize, 2, vulkan_data).try_into().unwrap());
                        self.registers[i_format.rd as usize] = value as u64;
                    }
                    0b110 => {
                        //lwu
                        let value = u32::from_le_bytes(self.get_bytes(address as usize, 4, vulkan_data).try_into().unwrap());
                        self.registers[i_format.rd as usize] = value as u64;
                    }
                    _ => {self.unimplemented_instruction(instruction);}
                }

            }
            opcodes::STORE =>{
                let s_format = format::S::new(instruction);
                let address = self.registers[s_format.rs1 as usize].signed_offset(s_format.imm);
                match s_format.funct3{
                    0b000 => {
                        self.set_bytes(address as usize, 1, self.registers[s_format.rs2 as usize].to_le_bytes().try_into().unwrap(), vulkan_data);
                    }
                    0b001 => {
                        self.set_bytes(address as usize, 2, self.registers[s_format.rs2 as usize].to_le_bytes().try_into().unwrap(), vulkan_data);

                    },
                    0b010 => {
                        self.set_bytes(address as usize, 4, self.registers[s_format.rs2 as usize].to_le_bytes().try_into().unwrap(), vulkan_data);

                    },
                    0b011 => {
                        //SD
                        self.set_bytes(address as usize, 8, self.registers[s_format.rs2 as usize].to_le_bytes().try_into().unwrap(), vulkan_data);

                    },
                    _ => {self.unimplemented_instruction(instruction);}
                }
            }
            opcodes::W_ARITHEMETIC => {
                let r_format = format::R::new(instruction);
                match r_format.funct3{
                    0b000 => {
                        match r_format.funct7{
                            0b0000000 => {
                                //ADDW
                                self.registers[r_format.rd as usize] = self.registers[r_format.rs1 as usize].wrapping_add(self.registers[r_format.rs2 as usize]) as u32 as u64;
                            }
                            0b0100000 => {
                                //SUBW
                                self.registers[r_format.rd as usize] = self.registers[r_format.rs1 as usize].wrapping_sub(self.registers[r_format.rs2 as usize]) as u32 as u64;
                            }
                            _ => {self.unimplemented_instruction(instruction);}

                        }
                    }
                    0b001 => {
                        //SLLW
                        self.registers[r_format.rd as usize] = ((self.registers[r_format.rs1 as usize] as u32) << (self.registers[r_format.rs2 as usize] & 0b111111) as u32) as u64;
                    }
                    0b101 => {
                        match r_format.funct7{
                            0b0000000 => {
                                //SRLW
                                self.registers[r_format.rd as usize] = ((self.registers[r_format.rs1 as usize] as u32) >> (self.registers[r_format.rs2 as usize] & 0b111111) as u32) as u64;

                            }
                            0b0100000 => {
                                //SRAW
                                let rs1 = i32::from_le_bytes(self.registers[r_format.rs1 as usize].to_le_bytes()[..4].try_into().unwrap());
                                let rs2 = i32::from_le_bytes((self.registers[r_format.rs2 as usize] & 0b111111).to_le_bytes()[..4].try_into().unwrap());
                                self.registers[r_format.rd as usize] = u32::from_le_bytes((rs1 >> rs2).to_le_bytes()) as u64;

                            }
                            _ => {self.unimplemented_instruction(instruction);}
                        }
                    }
                    _ => {self.unimplemented_instruction(instruction)}
                }
            }
            opcodes::ARITHMETIC => {
                let r_format = format::R::new(instruction);
                match r_format.funct3 {
                    0b000 => {
                        match r_format.funct7{
                            0b0000000 => {
                                // add
                                self.registers[r_format.rd as usize] = self.registers[r_format.rs1 as usize].wrapping_add(self.registers[r_format.rs2 as usize]);
                            }
                            0b0100000 => {
                                //SUB
                                self.registers[r_format.rd as usize] = self.registers[r_format.rs1 as usize].wrapping_sub(self.registers[r_format.rs2 as usize]);
                            }
                            _ => {self.unimplemented_instruction(instruction);}
                        }
                    }
                    0b001 => {
                        //SLL
                        self.registers[r_format.rd as usize] = self.registers[r_format.rs1 as usize] << (self.registers[r_format.rs2 as usize] & 0b111111);

                    }
                    0b010 => {
                        //SLT
                        self.registers[r_format.rd as usize] = if i64::from_le_bytes(self.registers[r_format.rs1 as usize].to_le_bytes()) < i64::from_le_bytes(self.registers[r_format.rs2 as usize].to_le_bytes()){
                            1
                        }else{
                            0
                        }
                    }
                    0b11 => {
                        //SLTU
                        self.registers[r_format.rd as usize] = if self.registers[r_format.rs1 as usize] < self.registers[r_format.rs2 as usize]{
                            1
                        }else{
                            0
                        }
                    }
                    0b100 => {
                        //XOR
                        self.registers[r_format.rd as usize] = self.registers[r_format.rs1 as usize] ^ self.registers[r_format.rs2 as usize];
                    }
                    0b101 => {
                        match r_format.funct7{
                            0b0000000 => {
                                //SRL
                                self.registers[r_format.rd as usize] = self.registers[r_format.rs1 as usize] >> (self.registers[r_format.rs2 as usize] & 0b111111);

                            }
                            0b0100000 => {
                                //SRA
                                self.registers[r_format.rd as usize] = u64::from_le_bytes((i64::from_le_bytes(self.registers[r_format.rs1 as usize].to_le_bytes()) >> i64::from_le_bytes((self.registers[r_format.rs2 as usize] & 0b111111).to_le_bytes())).to_le_bytes());

                            }
                            _ => {self.unimplemented_instruction(instruction);}
                        }
                    }
                    0b110 => {
                        //OR
                        self.registers[r_format.rd as usize] = self.registers[r_format.rs1 as usize] | self.registers[r_format.rs2 as usize];

                    }
                    0b111 => {
                        //AND
                        self.registers[r_format.rd as usize] = self.registers[r_format.rs1 as usize] & self.registers[r_format.rs2 as usize];

                    }
                    _ => {self.unimplemented_instruction(instruction);}
                }

            }
            opcodes::W_IMMEDIATE_ARITHMETIC => {
                let i_format = format::I::new(instruction);
                let shift_i_format = format::ShiftI::new(instruction);

                match i_format.funct3{
                    0b000 =>{
                        // ADDIW
                        //this is a little gross because we need to get the sign extension right
                        // self.registers[i_format.rd as usize] =
                        //     u64::from_le_bytes(
                        //         (i32::from_le_bytes(
                        //             (self.registers[i_format.rs1 as usize].to_le_bytes())
                        //                 .wrapping_add(i_format.imm)
                        //                 .to_le_bytes()[4..]
                        //                 .try_into().unwrap()) as i64)
                        //             .to_le_bytes());

                        let immediate = i32::from_le_bytes(i_format.imm.to_le_bytes()[..4].try_into().unwrap());
                        let rs1 = i32::from_le_bytes(self.registers[i_format.rs1 as usize].to_le_bytes()[..4].try_into().unwrap());
                        let output = (immediate.wrapping_add(rs1)) as i64;
                        self.registers[i_format.rd as usize] = u64::from_le_bytes(output.to_le_bytes());

                    }
                    0b001 => {
                        //SLLIW
                        self.registers[shift_i_format.rd as usize] = self.registers[shift_i_format.rs1 as usize] << shift_i_format.shamt;
                    }
                    0b101 => {
                        match shift_i_format.funct7{
                            0b0000000 =>{
                                //SRLIW
                                self.registers[shift_i_format.rd as usize] = self.registers[shift_i_format.rs1 as usize] << shift_i_format.shamt;
                            }
                            0b0100000 =>{
                                //SRAIW
                                self.registers[shift_i_format.rd as usize] = u64::from_le_bytes((i64::from_le_bytes(self.registers[shift_i_format.rs1 as usize].to_le_bytes()) << shift_i_format.shamt).to_le_bytes());

                            }
                            _ => {self.unimplemented_instruction(instruction)}
                        }
                    }
                    _ => {self.unimplemented_instruction(instruction);}
                }
            }
            opcodes::IMMEDIATE_ARITHMETIC => {
                let i_format = format::I::new(instruction);
                let shift_i_format = format::ShiftI::new(instruction);

                match i_format.funct3{
                    0b000 =>{
                        // addi
                        self.registers[i_format.rd as usize] = self.registers[i_format.rs1 as usize].signed_offset( i_format.imm);

                    }
                    0b010 => {
                        //SLTI
                        self.registers[i_format.rd as usize] = if i64::from_le_bytes(self.registers[i_format.rs1 as usize].to_le_bytes()) < i_format.imm{
                            1
                        }else{
                            0
                        }
                    }
                    0b011 => {
                        //SLTIU
                        self.registers[i_format.rd as usize] = if self.registers[i_format.rs1 as usize] < u64::from_le_bytes(i_format.imm.to_le_bytes()){
                            1
                        }else{
                            0
                        }
                    }
                    0b100 => {
                        //XORI
                        self.registers[i_format.rd as usize] = self.registers[i_format.rs1 as usize] ^ u64::from_le_bytes(i_format.imm.to_le_bytes());
                    }
                    0b110 => {
                        //ORI
                        self.registers[i_format.rd as usize] = self.registers[i_format.rs1 as usize] | u64::from_le_bytes(i_format.imm.to_le_bytes());
                    }
                    0b111 => {
                        //ANDI
                        self.registers[i_format.rd as usize] = self.registers[i_format.rs1 as usize] & u64::from_le_bytes(i_format.imm.to_le_bytes());
                    }
                    0b001 => {
                        //SLLI
                        self.registers[shift_i_format.rd as usize] = self.registers[shift_i_format.rs1 as usize] << shift_i_format.shamt;
                    }
                    0b101 => {
                        match shift_i_format.funct7{
                            0b0000000 => {
                                //SRLI
                                self.registers[shift_i_format.rd as usize] = self.registers[shift_i_format.rs1 as usize] >> shift_i_format.shamt;

                            }
                            0b0100000 => {
                                //SRAI
                                self.registers[shift_i_format.rd as usize] = u64::from_le_bytes((i64::from_le_bytes(self.registers[shift_i_format.rs1 as usize].to_le_bytes()) >> shift_i_format.shamt).to_le_bytes());

                            }
                            _ => {self.unimplemented_instruction(instruction);}
                        }
                    }
                    _ => {self.unimplemented_instruction(instruction);}
                }
            }
            opcodes::FENCE => {
                //Only one HART(core), don't think I need to do anything for FENCE
            }
            opcodes::ENVIRONMENT => {
                match (instruction & !0b1111111) >> 20{
                    0b0 => {
                        //ECALL
                        let stack_pointer = self.registers[2] as usize;
                        let system_call_type = self.get_bytes(stack_pointer, 8, vulkan_data).try_into().unwrap();
                        let system_call_type = u64::from_le_bytes(system_call_type);

                        match system_call_type{
                            1 =>{
                                let string_address = self.get_bytes(stack_pointer+8, 8, vulkan_data).try_into().unwrap();
                                let string_address = u64::from_le_bytes(string_address);

                                let mut string = vec![];

                                for i in (string_address as usize)..{
                                    let byte = self.get_bytes(i, 1, vulkan_data)[0];
                                    if byte == 0x0{
                                        break;
                                    }
                                    string.push(byte);

                                }

                                let string = CString::new(string).unwrap();
                                let string = string.to_string_lossy();
                                let mut arguments = vec![];
                                let num_arguments = u64::from_le_bytes(self.get_bytes(stack_pointer + 16, 8, vulkan_data).try_into().unwrap()) as usize;
                                let argument_address = u64::from_le_bytes(self.get_bytes(stack_pointer + 24, 8, vulkan_data).try_into().unwrap()) as usize;
                                for i in 0..num_arguments{
                                    arguments.push(i64::from_le_bytes(self.get_bytes(argument_address + i*8, 8, vulkan_data).try_into().unwrap()));
                                }

                                println!("{:}{:?}",string,arguments);

                            }
                            _ => println!("unknown system call")
                        }
                    }
                    0b1 => {
                        //EBREAK
                        println!("EBREAK at {:X}", self.program_counter);
                        return ProcessorResult::Break;
                    }
                    _ => {self.unimplemented_instruction(instruction);}
                }
            }
            opcodes::ZERO => {}
            _ => {
                self.unimplemented_instruction(instruction);
            }
        }

        self.registers[0] = 0; //TODO: Ignore writes to x0 rather than lazily zeroing it, maybe do some benchmarks to see if it matters


        if !program_counter_handled{
            self.program_counter += 4;
        }

        return ProcessorResult::Continue;
        // println!("Operation took {:?}", time_elapsed);
    }


}

trait SignedOffset{
    fn signed_offset(self, offset: i64) -> Self;
}

impl SignedOffset for u64{
    fn signed_offset(self, offset: i64) -> Self {
        return if offset >= 0 {
            self.wrapping_add(offset as u64)
        } else {
            self.wrapping_sub(-offset as u64)
        }
    }
}

