import torch
from pathlib import Path
import json
import sys

import dump
from model import Transformer, ModelArgs
import tokenizer

def load_model(model_dir, tokenizer_path):
    tok = tokenizer.Tokenizer(model_path=tokenizer_path)
    checkpoints = sorted(Path(model_dir).glob("*.pth"))
    if len(checkpoints) == 0:
        raise ValueError(f"No checkpoint files found in {model_dir}")
    
    weights = [torch.load(filename, map_location="cpu") for filename in checkpoints]
    with open(Path(model_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    
    model_args: ModelArgs = ModelArgs(
        max_batch_size=1,
        **params,
    )
    model_args.vocab_size = tok.n_words
    model = Transformer(model_args)
    model.load_state_dict(concat_weights(weights), strict=False)
    model.max_seq_len = model.tok_embeddings.weight.shape[0]
    print('Loaded model')

    return model


def concat_weights(models):
  def convert(name) -> torch.Tensor:
    disk_tensors = [model[name] for model in models]
    if len(disk_tensors) == 1:
      return disk_tensors[0]
    if len(disk_tensors[0].shape) == 1:
      return disk_tensors[0]
    if name.startswith('tok_embeddings.') or name.endswith('.attention.wo.weight') or name.endswith('.feed_forward.w2.weight'):
      axis = 1
    else:
      axis = 0
    first, rest = disk_tensors[0], disk_tensors[1:]
    return first.cat(*rest, dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("You must provide the model_dir and tok_path as command line parameters")

    model_dir = sys.argv[1]
    tokenizer_path = sys.argv[2]

    try:
        with torch.no_grad():
          llama = load_model(model_dir, tokenizer_path)

          tokens = torch.tensor([0, 2, 1])
          out = llama(tokens.unsqueeze(0), 0)

          print(out[0, :3, :10].numpy())
    except Exception as e:
        print(f"An error occurred: {e}")
