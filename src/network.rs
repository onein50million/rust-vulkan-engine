use std::convert::TryInto;
use bincode::config::Configuration;
use crate::game::sendable::{GameObject, World};
use serde::{Serialize, Deserialize};
use crate::game::GameObjectKey;
use crate::support::Inputs;


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


#[derive(Copy, Clone, Debug)]
pub enum ClientState{
    Disconnected,
    ConnectionAwaiting,
    Connected,
}



#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Packet{
    RequestConnect{username: String},
    RequestAccepted,
    RequestDenied,
    RequestGameWorld,
    GameWorld(World),
    Input(Inputs),
    GameObject{key: GameObjectKey, object: GameObject}
}

impl Packet{
    pub fn to_bytes(&self) -> Vec<u8>{
        let config = bincode::config::standard().with_big_endian();

        bincode::serde::encode_to_vec(self,config).unwrap()
    }
    pub fn from_bytes(bytes: &[u8]) -> Option<Self>{
        let config = bincode::config::standard().with_big_endian();

        match bincode::serde::decode_from_slice(bytes, config){
            Err(_) => {
                println!("Invalid packet");
                None },
            Ok(result) => Some(result.0)
        }
    }
}