import pathlib
import torch
import numpy as np

import model

def save_scalar(s, name, path):
    s = np.array([1.0, float(s)]).astype(np.float32)
    np.save(pathlib.Path(path, f'{name}.npy'), s)

def save_tensor(tensor, name, path):
    tensor_numpy = tensor.numpy()
    tensor_dims = np.array(tensor_numpy.shape)
    tensor_values = tensor_numpy.flatten()
    tensor_to_save = np.concatenate((tensor_dims, tensor_values)).astype(np.float32)
    np.save(pathlib.Path(path, f'{name}.npy'), tensor_to_save)

def save_linear(linear, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(linear.weight.t(), 'weight', path) # PyTorch and Tinygrad strangely transpose linear weights so reverse that
    if linear.bias is not None:
        save_tensor(linear.bias, 'bias', path)



def save_rmsnorm(norm, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(norm.weight, 'weight', path)
    save_scalar(norm.eps, 'eps', path)

def save_attention(attention, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(attention.wq, pathlib.Path(path, 'wq'))
    save_linear(attention.wk, pathlib.Path(path, 'wk'))
    save_linear(attention.wv, pathlib.Path(path, 'wv'))
    save_linear(attention.wo, pathlib.Path(path, 'wo'))
    n_kv_head = attention.n_kv_heads
    n_head = n_kv_head * attention.n_rep
    save_scalar(n_head, "n_head", path)
    save_scalar(n_kv_head, "n_kv_head", path)

def save_feedforward(feed_forward, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(feed_forward.w1, pathlib.Path(path, 'w1'))
    save_linear(feed_forward.w2, pathlib.Path(path, 'w2'))
    save_linear(feed_forward.w3, pathlib.Path(path, 'w3'))

def save_embedding(embedding, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(embedding.weight, 'weight', path)

def save_transformer_block(transformer_block, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_attention(transformer_block.attention, pathlib.Path(path, 'attention'))
    save_feedforward(transformer_block.feed_forward, pathlib.Path(path, 'feedforward'))
    save_rmsnorm(transformer_block.attention_norm, pathlib.Path(path, 'attention_norm'))
    save_rmsnorm(transformer_block.ffn_norm, pathlib.Path(path, 'ffn_norm'))

def save_transformer(transformer, path):
    with torch.no_grad():
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        save_scalar(len(transformer.layers), 'n_layer', path)
        for idx, layer in enumerate(transformer.layers):
            save_transformer_block(layer, pathlib.Path(path, f'layer{idx}'))
        save_rmsnorm(transformer.norm, pathlib.Path(path, 'norm'))
        save_embedding(transformer.tok_embeddings, pathlib.Path(path, 'tok_embeddings'))
        save_linear(transformer.output, pathlib.Path(path, 'output'))
        save_scalar(10000.0, 'theta', path)
        save_scalar(transformer.params.max_seq_len, 'n_ctx', path)
        save_scalar(transformer.params.multiple_of, 'multiple_of', path)
        if transformer.params.ffn_dim_multiplier is not None:
            save_scalar(transformer.params.ffn_dim_multiplier, 'ffn_dim_multiplier', path)
        #save_tensor(transformer.freqs_cis, 'freqs_cis', path)
