use rust_vulkan_engine::network::{ClientState, Packet};
use rust_vulkan_engine::support::Inputs;
use std::collections::HashMap;
use std::net::{SocketAddr, UdpSocket};
use std::time::Instant;

pub struct Client {
    address: SocketAddr,
    state: ClientState,
    username: String,
    pub(crate) inputs: Inputs,
}
impl Client {
    fn new(address: SocketAddr) -> Self {
        let out = Self {
            address,
            state: ClientState::Connected,
            username: "".to_string(),
            inputs: Inputs::new(),
        };
        out
    }
    fn process_packet(&mut self, packet: Packet, socket: &UdpSocket) {
        match packet {
            Packet::RequestConnect { username: _ } => {
                self.state = ClientState::Connected;
                let send_data = Packet::RequestAccepted;

                let send_data = send_data.to_bytes();
                socket.send_to(&send_data, self.address).unwrap();
            }
            Packet::RequestAccepted => {}
            Packet::RequestDenied => {}
            Packet::RequestGameWorld => {}
            Packet::Input(_) => {}
        }
    }
}

struct Server {
    clients: HashMap<SocketAddr, Client>,
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
    fn process(&mut self) {
        let mut buffer = [0; 1024];
        while let Ok((num_bytes, source_address)) = self.socket.recv_from(&mut buffer) {
            let unprocessed_datagram = &mut buffer[..num_bytes];
            match Packet::from_bytes(unprocessed_datagram) {
                None => {}
                Some(packet) => {
                    self.clients
                        .entry(source_address)
                        .or_insert(Client::new(source_address))
                        .process_packet(packet, &self.socket);
                }
            }
        }
    }
}

fn main() {
    let mut server = Server::new();

    loop {
        server.process();
    }
}
