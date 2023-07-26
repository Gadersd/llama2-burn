import torch
import dump
import model
from model import Transformer, ModelArgs

if __name__ == "__main__":
    n_vocab = 10
    n_ctx = 15
    n_state = 8
    multiple_of = 3
    n_head = 4
    n_kv_head = 2
    n_layer = 3
    norm_eps = 1e-6
    max_batch_size = 1

    model_args = ModelArgs(dim=n_state, n_layers=n_layer, n_heads=n_head, n_kv_heads=n_kv_head,
                           vocab_size=n_vocab, multiple_of=multiple_of, norm_eps=norm_eps, 
                           max_batch_size=max_batch_size)

    llama = Transformer(model_args)

    with torch.no_grad():
        tokens = torch.tensor([0, 2, 1], dtype=torch.int32).unsqueeze(0)
        output = llama(tokens, 0)
        print(f'Test input {tokens.numpy()}')
        print(f'Test output {output.numpy()}')

        print('Dumping test model...')
        dump.save_transformer(llama, "params")
        print('Dump saved in params folder.')