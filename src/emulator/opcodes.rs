//opcodes go here

pub(crate) const LUI: u32 = 0b110111;
pub(crate) const AUIPC: u32 = 0b0010111;
pub(crate) const JAL: u32 = 0b1101111;
pub(crate) const JALR: u32 = 0b1100111;
pub(crate) const BRANCH: u32 = 0b1100011;

pub(crate) const LOAD: u32 = 0b0000011;
pub(crate) const STORE: u32 = 0b0100011;

pub(crate) const IMMEDIATE_ARITHMETIC: u32 = 0b0010011;
pub(crate) const ARITHMETIC: u32 = 0b0110011;
pub(crate) const W_IMMEDIATE_ARITHMETIC: u32 = 0b0011011;
pub(crate) const W_ARITHEMETIC: u32 = 0b0111011;


pub(crate) const FENCE: u32 = 0b0001111;
pub(crate) const ENVIRONMENT: u32 = 0b1110011;

pub(crate) const ZERO: u32 = 0b0;


