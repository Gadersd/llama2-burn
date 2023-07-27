use llama::model::*;
use llama::token::LlamaTokenizer;

use burn::{
    config::Config, 
    module::Module, 
    tensor::{
        self, 
        backend::{self, Backend},
        Data, 
        Tensor,
        Int, 
        Float, 
    },
};

fn test_tokenizer(tokenizer: &LlamaTokenizer) {
    println!("Vocab size: {}", tokenizer.vocab_size(false));

    let test_str = "Hello, I am Llama2!";
    let encoded = tokenizer.encode(test_str, true, true);
    let decoded = tokenizer.decode(&encoded, false);

    println!("Test string: {}", test_str);
    println!("Encoded tokens: {:?}", encoded);
    println!("Decoded string: {}", decoded);
}

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

use std::env;
use std::io;
use std::process;

fn main() {
    let device = burn_tch::TchDevice::Cpu;
    type Backend = burn_tch::TchBackend<Elem>;

    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <tokenizer_filepath> <dump_path>", args[0]);
        process::exit(1);
    }
    
    let tokenizer_filepath = &args[1];
    let dump_path = &args[2];

    let tokenizer = match LlamaTokenizer::new(tokenizer_filepath) {
        Ok(tokenizer) => tokenizer,
        Err(e) => {
            eprintln!("Failed to create tokenizer: {:?}", e);
            process::exit(1);
        }
    };

    test_tokenizer(&tokenizer);

    let (llama, llama_config): (Llama::<Backend>, LlamaConfig) = match load_llama_dump(dump_path) {
        Ok((llama, llama_config)) => (llama, llama_config),
        Err(e) => {
            eprintln!("Failed to load llama dump: {:?}", e);
            process::exit(1);
        }
    };

    let tokens = Tensor::from_ints([0, 2, 1]);
    let out = llama.forward(tokens.clone().unsqueeze());

    println!("Tokens input: {:?}", tokens.into_data());
    println!("Model output: {:?}", out.slice([0..1, 0..3, 0..10]).into_data());
}
