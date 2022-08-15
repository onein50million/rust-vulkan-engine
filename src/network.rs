use bincode::{Decode, Encode};
use nalgebra::{Vector2, Vector3};

use crate::{game::{GameTick, PlayerKey, PlayerState}, support::Inputs};
/*
Client requests Connection
Server Accepts connection
client requests world
server sends world
loop {
    client sends inputs
    server sends game object positions
}


 */

// #[derive(Encode, Decode, Copy, Clone)]
// pub(crate) enum NetworkResult{
//     Success,
//     GenericFail
// }

pub const NETWORK_TICKRATE: f64 = 100.0;

#[derive(Copy, Clone, Debug)]
pub enum ClientState {
    Disconnected,
    ConnectionAwaiting,
    Connected,
    TimedOut,
}

#[derive(Encode, Decode, Clone, Debug)]
pub enum ClientToServerPacket {
    RequestConnect {
        username: String,
    },
    Input {
        inputs: Inputs,
        tick_sent: f64,
    },
}

impl ClientToServerPacket {
    pub fn to_bytes(&self) -> Vec<u8> {
        let config = bincode::config::standard().with_big_endian();

        bincode::encode_to_vec(self, config).unwrap()
    }
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let config = bincode::config::standard().with_big_endian();

        match bincode::decode_from_slice(bytes, config) {
            Err(_) => {
                println!("Invalid packet");
                None
            }
            Ok(result) => Some(result.0),
        }
    }
}
#[derive(Encode, Decode, Clone, Debug)]
pub struct NetworkPlayer {
    name: String,
}

#[derive(Encode, Decode, Clone, Debug)]
pub enum ServerToClientPacket {
    RequestAccepted(GameTick, PlayerKey, usize),
    RequestDenied,
    PlayerUpdate {
        key: PlayerKey,
        player_state: PlayerState,
        last_client_input: f64,
    },
}
impl ServerToClientPacket {
    pub fn to_bytes(&self) -> Vec<u8> {
        let config = bincode::config::standard().with_big_endian();

        let out = bincode::encode_to_vec(self, config).unwrap();
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let config = bincode::config::standard().with_big_endian();

        match bincode::decode_from_slice(bytes, config) {
            Err(_) => {
                println!("Invalid packet");
                None
            }
            Ok(result) => Some(result.0),
        }
    }
}
