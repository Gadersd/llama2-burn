# Llama2-burn Project

## Overview

This project provides a port of Meta's large language model, Llama2, to the Burn, a Rust deep learning framework. Specifically, the project hosts Python and Rust scripts necessary for the conversion, loading, and verification of weights from the Llama2 model into parameters compatible with Burn's framework.

## Pre-requisites

You will need to obtain the Llama2 model files to use this project. These can be downloaded directly from Meta or via Hugging Face's Model Hub.

## Project Structure

The directory structure of the project is as follows:

```
.
├── Cargo.lock
├── Cargo.toml
├── LICENSE
├── README.md
├── llama-py
│   ├── dump.py
│   ├── dump_model.py
│   ├── dump_test.py
│   ├── model.py
│   ├── requirements.txt
│   ├── test.py
│   ├── test_tokenizer.py
│   └── tokenizer.py
└── src
    ├── bin
    │   ├── convert
    │   │   └── main.rs
    │   ├── sample
    │   │   └── main.rs
    │   └── test
    │       └── main.rs
    ├── helper.rs
    ├── lib.rs
    ├── model.rs
    └── token.rs
```

## Usage

Follow the steps below to go from the downloaded Llama2 model files to dumping, conversion, and running in Rust:

### Step 1: Loading and Testing with Python Scripts

Inside the `llama-py` folder, you will find the necessary Python scripts. Here, you will primarily use `test.py`, `dump_model.py`, and `test_tokenizer.py`.

1. **Test the Model**: Run the `test.py` script to load the model and verify it with a short prompt. If the output is gibberish, then there might be an issue with the model. Execute this script using the command:
```
python3 test.py <model_dir> <tokenizer_path>
```
Example: `python3 test.py llama2-7b-chat tokenizer.model`

2. **Dump the Model Weights**: Run the `dump_model.py` script to load the model and dump the weights into the `params` folder ready for loading in Rust.  Execute this script using the command:
```
python3 dump_model.py <model_dir> <tokenizer_path>
```
Example: `python3 dump_model.py llama2-7b-chat tokenizer.model`

3. **Test the Tokenizer**: Finally, run the `test_tokenizer.py` script to load the tokenizer.model file and verify an example encoding and decoding. This script should be run in the same directory as the tokenizer file. Execute this script using the command: 
```
python3 test_tokenizer.py
```

### Step 2: Conversion and Running with Rust Binaries

Inside the 'src/bin' folder, you will find Rust binaries: `convert`, `sample`, and `test`.

1. **Converting Dumped Weights**: The `convert` binary converts dumped weights into burn's model format. It saves them for further use. Execute this using the following command: 
```
cargo run --bin convert <dump_path> <burn_model_name>
```
Example: `cargo run --release --bin convert params llama2-7b-chat`

2. **Testing Weights And Rust Inference Code**: The `test` binary loads the dumped weights and tests an example prompt to examine if the model weights and rust inference code produce sensible output. It is a companion to `test.py`. Execute this using the command:
```
cargo run --bin test <tokenizer_filepath> <dump_path>
```
Example: `cargo run --release --bin test tokenizer.model params`

3. **Sampling Text**: The `sample` binary loads the converted burn model file and generates a sample output based on an input prompt. The model can run on either the cpu or gpu. Execute this using the following command: 
```
cargo run --bin sample <model_name> <tokenizer_filepath> <prompt> <n_tokens>
```
Example: 
```
#export TORCH_CUDA_VERSION=cu113 # if running on gpu
cargo run --release --bin sample llama2-7b-chat tokenizer.model "Hello, I am " 10 cpu
```

## Note

Loading and converting weights occur on your CPU. Please ensure your CPU has enough RAM to hold the model and some extra resources for smooth functioning. Be patient as the process may take a while.

## Contribution

Contributions to improve this project are most welcome. Feel free to clone the repository, test the conversions, and submit a pull request for any enhancements, bug fixes, or other improvements.

## License

This project is licensed as specified in the LICENSE file.

---

Thanks for testing out the Llama2-Burn project! Happy coding!
