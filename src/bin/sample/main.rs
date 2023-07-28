use llama::model::*;
use llama::token::LlamaTokenizer;

use num_traits::cast::ToPrimitive;
use std::error::Error;

use burn_wgpu::{WgpuBackend, WgpuDevice, AutoGraphicsApi};

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

use burn::record::{self, Recorder, DefaultRecorder};

fn load_llama<B: Backend>(model_name: &str) -> Result<(Llama<B>, LlamaConfig), Box<dyn Error>> {
    let config = LlamaConfig::load(&format!("{model_name}.cfg"))?;
    let llama = load_llama_model_file(&config, model_name)?;

    Ok( (llama, config) )
}

fn load_llama_model_file<B: Backend>(config: &LlamaConfig, filename: &str) -> Result<Llama<B>, record::RecorderError> {
    DefaultRecorder::new()
    .load(filename.into())
    .map(|record| config.init().load_record(record))
}

fn convert_llama_dump_to_model<B: Backend>(dump_path: &str, model_name: &str) -> Result<(), Box<dyn Error>> {
    let (llama, llama_config): (Llama::<B>, LlamaConfig) = load_llama_dump(dump_path)?;

    save_llama_model_file(llama, model_name)?;
    llama_config.save(&format!("{model_name}.cfg"))?;

    Ok( () )
}

fn save_llama_model_file<B: Backend>(llama: Llama<B>, name: &str) -> Result<(), record::RecorderError> {
    DefaultRecorder::new()
    .record(
        llama.into_record(),
        name.into(),
    )
}

fn sample_llama<B: Backend>(llama: &Llama<B>, tokenizer: &LlamaTokenizer, prompt: &str, n_tokens: usize) -> String {
    let device = llama.devices()[0].clone();

    let mut tokens = tokenizer.encode(prompt, true, false);
    let mut text = String::new();

    for i in 0..n_tokens {
        let token_tensor = Tensor::from_ints(
            Data::from_usize(Data::new(tokens.iter().map(|&t| t as usize).collect(), [tokens.len()].into()))
        ).unsqueeze::<2>()
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

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

use std::env;
use std::io;
use std::process;

fn main() {
    type Backend = WgpuBackend<AutoGraphicsApi, Elem, i32>;
    let device = WgpuDevice::BestAvailable;

    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: {} <model_name> <tokenizer_filepath> <prompt> <n_tokens>", args[0]);
        process::exit(1);
    }

    let model_name = &args[1];
    let tokenizer_filepath = &args[2];
    let prompt = &args[3];
    let n_tokens: usize = args[4].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid number of tokens");
        process::exit(1);
    });

    let tokenizer = match LlamaTokenizer::new(tokenizer_filepath) {
        Ok(tokenizer) => tokenizer,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {:?}", e);
            process::exit(1);
        }
    };

    let (llama, llama_config): (Llama::<Backend>, LlamaConfig) = match load_llama(model_name) {
        Ok((llama, llama_config)) => (llama, llama_config),
        Err(e) => {
            eprintln!("Failed to load llama model: {:?}", e);
            process::exit(1);
        }
    };

    println!("Llama config: {:?}", llama_config);

    let llama = llama.to_device(&device);
    let sampled = sample_llama(&llama, &tokenizer, prompt, n_tokens);

    println!("Prompt: {}", prompt);
    println!("Output: {}{}", prompt, sampled);
}
