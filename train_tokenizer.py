
import os
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

print("Training BPE tokenizer...")
path = "input.txt"
tokenizer.train(
    files=[path], 
    vocab_size=1000, 
    min_frequency=2, 
    special_tokens=["<|endoftext|>"]
)
os.makedirs("shakespeare_tokenizer", exist_ok=True)
tokenizer.save_model("shakespeare_tokenizer")
print("Tokenizer trained and saved to 'shakespeare_tokenizer/' folder!")
sample_text = "ROMEO: What light through yonder window breaks?"
encoded = tokenizer.encode(sample_text)
print("\n--- Testing Tokenizer ---")
print(f"Original Text: {sample_text}")
print(f"Tokens: {encoded.tokens}")
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded Back: {decoded}")