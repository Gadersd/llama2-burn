use llama::model::*;
use llama::token::LlamaTokenizer;

use num_traits::cast::ToPrimitive;

use burn_tch::{TchBackend, TchDevice};

use burn::{
    config::Config,
    module::Module,
    tensor::{
        self,
        backend::{self, Backend},
        Data, Float, Int, Tensor,
    },
};

fn sample_llama<B: Backend>(
    llama: &Llama<B>,
    tokenizer: &LlamaTokenizer,
    prompt: &str,
    n_tokens: usize,
) -> String {
    let device = llama.devices()[0].clone();

    let mut tokens = tokenizer.encode(prompt, true, false);
    let mut text = String::new();

    for i in 0..n_tokens {
        let token_tensor = Tensor::from_ints(Data::from_usize(Data::new(
            tokens.iter().map(|&t| t as usize).collect(),
            [tokens.len()].into(),
        )))
        .unsqueeze::<2>()
        .to_device(&device);

        let out = llama.forward(token_tensor);

        let [n_batch, n_token, n_dict] = out.dims();
        let last_row: Tensor<B, 1> = out.slice([0..1, (n_token - 1)..n_token]).flatten(0, 2);

        let token_id = last_row.argmax(0).into_scalar().to_i64().unwrap();

        tokens.push(token_id);

        let token_text = tokenizer.decode(&[token_id], true);
        println!("{token_text}");

        text += &token_text;
    }

    text
}

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
    type Backend = TchBackend<f32>;

    // CPU is used for conversion
    // everyone who converts should be able to perform a simple test without needing a lot of GPU memory
    // so test on CPU
    let device = TchDevice::Cpu;

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

    let (llama, llama_config): (Llama<Backend>, LlamaConfig) =
        match load_llama_dump(dump_path, &device) {
            Ok((llama, llama_config)) => (llama, llama_config),
            Err(e) => {
                eprintln!("Failed to load llama dump: {:?}", e);
                process::exit(1);
            }
        };

    let test_prompt = "Hello, I am ";
    let sample = sample_llama(&llama, &tokenizer, test_prompt, 10);

    println!("Prompt: {}", test_prompt);
    println!("Sample: {}", sample);
    println!("Combined: {}{}", test_prompt, sample);
}
