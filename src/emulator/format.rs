#[derive(Debug)]
pub(crate) struct R{
    pub(crate) funct7: u32,
    pub(crate) rs2: u32,
    pub(crate) rs1: u32,
    pub(crate) funct3: u32,
    pub(crate) rd: u32,
    // opcode: u32, //opcode is shared
}
impl R{
    pub(crate) fn new(instruction: u32) -> Self{

        let rd = (instruction >> 7) & 0b11111;
        let funct3 = (instruction >> 12) & 0b111;
        let rs1 = (instruction >> 15) & 0b11111;
        let rs2 = (instruction >> 20) & 0b11111;
        let funct7= (instruction >> 25) & 0b1111111;

        return Self{
            funct7,
            rs2,
            rs1,
            funct3,
            rd,
        }
    }
}
#[derive(Debug)]
pub(crate) struct I{
    pub(crate) imm: i64,
    pub(crate) rs1: u32,
    pub(crate) funct3: u32,
    pub(crate) rd: u32,
}
impl I{
    pub(crate) fn new(instruction: u32) -> Self{

        let rd = (instruction >> 7) & 0b11111;
        let funct3 = (instruction >> 12) & 0b111;
        let rs1 = (instruction >> 15) & 0b11111;

        let imm = (i32::from_le_bytes(instruction.to_le_bytes()) >> 20) as i64;

        return Self{
            imm,
            rs1,
            funct3,
            rd,
        }
    }
}
#[derive(Debug)]
pub(crate) struct ShiftI{ //specialization of I-type for shifts
    pub(crate) funct7: u32,
    pub(crate) shamt: u32,
    pub(crate) rs1: u32,
    pub(crate) funct3: u32,
    pub(crate) rd: u32,
}
impl ShiftI{
    pub(crate) fn new(instruction: u32) -> Self{

        let rd = (instruction >> 7) & 0b11111;
        let funct3 = (instruction >> 12) & 0b111;
        let rs1 = (instruction >> 15) & 0b11111;
        let shamt = (instruction >> 20) & 0b11111;
        let funct7 = instruction >> 25;

        return Self{
            funct7,
            shamt,
            rs1,
            funct3,
            rd,
        }
    }
}
#[derive(Debug)]
pub(crate) struct S{
    pub(crate) imm: i64,
    pub(crate) rs2: u32,
    pub(crate) rs1: u32,
    pub(crate) funct3: u32,
}
impl S{
    pub(crate) fn new(instruction: u32) -> Self{

        let bit11_5 = (i32::from_le_bytes(instruction.to_le_bytes()) as i64 >> 20) & !0b1_1111;
        let bit4_0 = i32::from_le_bytes(((instruction & 0b1111_1000_0000) >> 7).to_le_bytes()) as i64;

        let imm = bit11_5 | bit4_0;

        let rs2 = (instruction >> 20) & 0b11111;
        let rs1 = (instruction >> 15) & 0b1_1111;
        let funct3 = (instruction >> 12) & 0b111;

        return Self{
            imm,
            rs2,
            rs1,
            funct3,
        }
    }
}

#[derive(Debug)]
pub(crate) struct B{
    pub(crate) imm: i64,
    pub(crate) rs2: u32,
    pub(crate) rs1: u32,
    pub(crate) funct3: u32,

}
impl B{
    pub(crate) fn new(instruction: u32) -> Self{

        let bit12 = u32::from_le_bytes((i32::from_le_bytes((instruction & 0b1000_0000_0000_0000_0000_0000_0000_0000).to_le_bytes()) >> 19).to_le_bytes());
        let bit10_5 = (instruction & 0x7E_00_00_00) >> 20;
        let bit4_1 = (instruction & 0xF_00) >> 7;
        let bit11 = (instruction & 0b1000_0000) << 4;

        let imm = i32::from_le_bytes((bit12| bit11 | bit10_5 | bit4_1).to_le_bytes()) as i64;
        let rs2 = (instruction >> 20) & 0b11111;
        let rs1 = (instruction >> 15) & 0b1_1111;
        let funct3 = (instruction >> 12) & 0b111;

        return Self{
            imm,
            rs2,
            rs1,
            funct3
        }
    }
}

#[derive(Debug)]
pub(crate) struct U{
    pub(crate) imm: i64,
    pub(crate) rd: u32,
}
impl U{
    pub(crate) fn new(instruction: u32) -> Self{

        let imm = i32::from_le_bytes((instruction & 0xFF_FF_F0_00).to_le_bytes()) as i64;
        let rd = (instruction >> 7) & 0b11111;

        return Self{
            imm,
            rd
        }
    }
}

#[derive(Debug)]
pub(crate) struct J{
    pub(crate) imm: i64,
    pub(crate) rd: u32,
}
impl J{
    pub(crate) fn new(instruction: u32) -> Self{

        let bit20 = u32:: from_le_bytes((i32::from_le_bytes((instruction & 0b1000_0000_0000_0000_0000_0000_0000_0000).to_le_bytes()) >> 11).to_le_bytes());
        let bit10_1 = (instruction & 0b0111_1111_1110_0000_0000_0000_0000_0000) >> 20;
        let bit11 = (instruction & 0b0001_0000_0000_0000_0000_0000) >> 9;
        let bit19_12 = (instruction & 0b0000_0000_0001_1111_1110_0000_0000_0000) >> 1 ;

        let imm = i32::from_le_bytes((bit20 | bit19_12 | bit11 | bit10_1).to_le_bytes()) as i64;
        let rd = (instruction >> 7) & 0b11111;

        return Self{
            imm,
            rd
        }
    }
}