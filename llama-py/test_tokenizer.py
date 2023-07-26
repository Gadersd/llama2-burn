import tokenizer
    
if __name__ == "__main__":
    tok = tokenizer.Tokenizer("tokenizer.model")

    test_str = "Hello, I am Llama2!"
    encoded = tok.encode(test_str, True, True)
    decoded = tok.decode(encoded)

    print(f"Test string: {test_str}")
    print(f"Encoded tokens: {encoded}")
    print(f"Decoded string: {decoded}")