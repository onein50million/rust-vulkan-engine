use tokio;
use sha3::Digest;

#[derive(serde::Deserialize)]
struct Result{
    input: String,
}

#[derive(serde::Deserialize)]
struct Response {
    result: Vec<Result>
}

#[derive(serde::Serialize)]
struct RPCParams {
    to: String,
}

#[derive(serde::Serialize)]
struct RPCRequest{
    jsonrpc:String,
    method: String,
    params: Vec<RPCParams>,
    id: u64,
}


#[tokio::main]
pub(crate) async fn getState(){
    println!("decode:{:?}", decode_hex("FFFFFF"));
    println!("encode:{:?}", encode_hex(vec![15,240, 69, 33, 255]));

    const NFTAddress: &str = "0xc31f33f941E5419fc593D9C376C030AF592e06C7";

    let client = reqwest::Client::new();
    let explorer_url = format!("\
        https://api-testnet.polygonscan.com/api\
        ?module=account\
        &action=txlist\
        &address={:}\
        &startblock=0\
        &endblock=99999999\
        &sort=desc\
        &apikey=YourApiKeyToken", NFTAddress);

    let rpc_url = "https://rpc-mumbai.matic.today";

    let rpc_response = client.post(rpc_url)
        .json(&RPCRequest{
            jsonrpc: String::from("2.0"),
            method: String::from("eth_call"),
            params: vec![],
            id: 1
        });

    let explorer_response = reqwest::get(explorer_url).await.unwrap().json::<Response>().await.unwrap();
    explorer_response.result.into_iter().for_each(|transaction| {
        let state = &decode_hex(&transaction.input[2..])[4+32..];
        println!("length: {:}", state.len());
        let hashed = sha3::Keccak256::digest(state);
        println!("hashed {:X?}", hashed);
    });

}

fn decode_hex(string:&str) -> Vec<u8>{
    let mut output = vec![];
    let char_array:Vec<char> = string.chars().collect();
    for i in (0..char_array.len()-1).step_by(2){
        output.push((16*char_array[i].to_digit(16).unwrap() + char_array[i+1].to_digit(16).unwrap()) as u8);
    }
    return output;
}

fn encode_hex(data: Vec<u8>) -> String{
    let mut output= String::new();
    for byte in data{
        let first_char = char::from_digit((byte / 16) as u32, 16).unwrap();
        let second_char = char::from_digit((byte - (byte/16)*16) as u32, 16).unwrap();
        output = format!("{:}{:}{:}",output,first_char,second_char);
    }
    return output;
}