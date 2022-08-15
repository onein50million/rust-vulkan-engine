use lyon_tessellation::geom::euclid::num::Zero;
use nalgebra::{Vector2, Vector3};
use rust_vulkan_engine::game::{GameObject, PlayerKey, PlayerObject, ServerGame, GameTick, PlayerState};
use rust_vulkan_engine::network::{ClientState, ClientToServerPacket, ServerToClientPacket, NETWORK_TICKRATE};
use rust_vulkan_engine::support::Inputs;
use std::collections::HashMap;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Instant, Duration};



const TIMEOUT_TIME: f64 = 4.0;
#[derive(Debug)]
pub struct ConnectedClient {
    address: SocketAddr,
    state: ClientState,
    username: String,
    player_key: Option<PlayerKey>,
    last_received_tick: GameTick,
}
impl ConnectedClient {
    fn new(address: SocketAddr) -> Self {
        let out = Self {
            address,
            state: ClientState::ConnectionAwaiting,
            username: "".to_string(),
            player_key: None,
            last_received_tick: GameTick::new(0),
        };
        out
    }
    fn process_packet(
        &mut self,
        packet: ClientToServerPacket,
        socket: &UdpSocket,
        game: &mut ServerGame,
    ) {
        match packet {
            ClientToServerPacket::RequestConnect { username } => {
                self.state = ClientState::Connected;
                self.username = username;
                let player_key = game.players.push(PlayerObject {
                    player_state: PlayerState{
                        position: Vector3::zeros(),
                        velocity: Vector3::zeros(),
                        is_jumping: false,
                    },
                    inputs: Inputs::new(),
                    last_input_time: None,
                });
                self.player_key = Some(player_key);
                let send_data = ServerToClientPacket::RequestAccepted(
                    game.current_tick,
                    player_key,
                    game.players.len(),
                );
                let send_data = send_data.to_bytes();
                socket.send_to(&send_data, self.address).unwrap();
                println!(
                    "Connection request from {:} with username {:}",
                    self.address, self.username
                )
            }
            ClientToServerPacket::Input {
                inputs,
                tick_sent,
            } => {
                if let Some(player_key) = self.player_key {
                    game.players[player_key].inputs = inputs;
                    game.players[player_key].last_input_time = Some(tick_sent);
                }
            }
        }
    }
}

struct Server {
    clients: HashMap<SocketAddr, ConnectedClient>,
    socket: UdpSocket,
    start_time: Instant,
}
impl Server {
    fn new() -> Self {
        let socket = UdpSocket::bind("127.0.0.1:2022").unwrap();
        socket
            .set_nonblocking(true)
            .expect("Failed to set socket as nonblocking");
        Self {
            clients: HashMap::new(),
            socket,
            start_time: Instant::now(),
        }
    }
    fn process(&mut self, game: &mut ServerGame) {
        let mut buffer = [0; 1024];
        while let Ok((num_bytes, source_address)) = self.socket.recv_from(&mut buffer) {
            let unprocessed_datagram = &mut buffer[..num_bytes];
            match ClientToServerPacket::from_bytes(unprocessed_datagram) {
                None => {}
                Some(packet) => {
                    let client = self.clients
                        .entry(source_address)
                        .or_insert(ConnectedClient::new(source_address));
                    client.last_received_tick = game.current_tick;
                    client.process_packet(packet, &self.socket, game);
                }
            }
        }
        for client in self.clients.values_mut() {

            if matches!(client.state, ClientState::TimedOut | ClientState::Disconnected){
                continue;
            }
            if game.current_tick.get() - client.last_received_tick.get() > (TIMEOUT_TIME * NETWORK_TICKRATE) as usize{
                client.state = ClientState::TimedOut;
                println!("Client timed out: {:}", client.address);
            }
            {
            for (player_key, player) in game.players.iter().enumerate(){
                let player_key = PlayerKey::new(player_key);
                let packet = ServerToClientPacket::PlayerUpdate {
                    key: player_key,
                    player_state: player.player_state,
                    last_client_input: player.last_input_time.unwrap_or(0.0),
                };
                match self.socket.send_to(&packet.to_bytes(), client.address) {
                    Ok(_) => {}
                    Err(error) => eprintln!("Socket error: {:}", error),
                }
            }
            }
        }
    }
}

fn main() {
    let mut server = Server::new();
    let mut game = ServerGame::new();

    let mut start = std::time::Instant::now();
    loop {
        while start.elapsed().as_secs_f64() > 1.0/NETWORK_TICKRATE{
            start += Duration::from_secs_f64(1.0/NETWORK_TICKRATE);

            server.process(&mut game);
            game.process(1.0 / NETWORK_TICKRATE);
        }
    }
}
