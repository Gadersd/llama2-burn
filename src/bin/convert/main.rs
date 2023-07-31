use llama::model::*;
use llama::token::LlamaTokenizer;

use num_traits::cast::ToPrimitive;
use std::error::Error;

use burn_tch::{TchBackend, TchDevice};

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

use burn::record::{self, Recorder, BinFileRecorder, HalfPrecisionSettings};

fn convert_llama_dump_to_model<B: Backend>(dump_path: &str, model_name: &str, device: &B::Device) -> Result<(), Box<dyn Error>> {
    let (llama, llama_config): (Llama::<B>, LlamaConfig) = load_llama_dump(dump_path, device)?;

    save_llama_model_file(llama, model_name)?;
    llama_config.save(&format!("{model_name}.cfg"))?;

    Ok( () )
}

fn save_llama_model_file<B: Backend>(llama: Llama<B>, name: &str) -> Result<(), record::RecorderError> {
    BinFileRecorder::<HalfPrecisionSettings>::new()
    .record(
        llama.into_record(),
        name.into(),
    )
}

fn test_tokenizer() {
    let tokenizer = LlamaTokenizer::new("tokenizer.model").unwrap();
    println!("Vocab size: {}", tokenizer.vocab_size(false));

    let test_str = "Hello, I am Llama2!";
    let encoded = tokenizer.encode(test_str, true, true);
    let decoded = tokenizer.decode(&encoded, false);

    println!("Test string: {}", test_str);
    println!("Encoded tokens: {:?}", encoded);
    println!("Decoded string: {}", decoded);
}

use std::env;
use std::io;
use std::process;

fn main() {
    type Backend = TchBackend<f32>;

    // might crash if lacking enough GPU memory so use CPU for conversion
    let device = TchDevice::Cpu;

    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <dump_path> <model_name>", args[0]);
        process::exit(1);
    }

    let dump_path = &args[1];
    let model_name = &args[2];

    if let Err(e) = convert_llama_dump_to_model::<Backend>(dump_path, model_name, &device) {
        eprintln!("Failed to convert llama dump to model: {:?}", e);
        process::exit(1);
    }

    println!("Successfully converted {} to {}", dump_path, model_name);
}
