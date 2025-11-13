import onnxruntime as ort
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def generate_text(session, tokenizer, prompt, max_new_tokens=50):
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    for _ in range(max_new_tokens):
        seq_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        
        if seq_len > 128:
            input_ids = input_ids[:, -128:]
            attention_mask = attention_mask[:, -128:]
        
        input_ids_np = input_ids.numpy().astype(np.int64)
        attention_mask_np = attention_mask.numpy().astype(np.int64)
        
        logits = session.run(None, {
            'input_ids': input_ids_np,
            'attention_mask': attention_mask_np
        })[0]
        
        next_token_logits = logits[0, -1, :] 

        next_token = np.argmax(next_token_logits)
        
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return generated_text


def main():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    session = ort.InferenceSession(
        "transformer_model.onnx",
        providers=['CPUExecutionProvider']
    )
    
    dataset_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    test = dataset_test.shuffle(seed=42).select(range(10))

    pad_token_id = tokenizer.pad_token_id
    def tokenize_data_labels(ex):
        curr_ex = tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
        labels = []
        for input_id_row in curr_ex["input_ids"]:
            shifted = input_id_row[1:] + [pad_token_id]
            processed_row = [-100 if token_id == pad_token_id else token_id for token_id in shifted]
            labels.append(processed_row)
        curr_ex["labels"] = labels
        return curr_ex
    
    test_tok = test.map(tokenize_data_labels, batched=True, remove_columns=["text"])
    test_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_loader = DataLoader(test_tok, batch_size=8)
    
    total_loss = 0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    for batch in test_loader:
        input_ids = batch["input_ids"].numpy().astype(np.int64)
        attention_mask = batch["attention_mask"].numpy().astype(np.int64)
        labels = batch["labels"]
        
        logits = session.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})[0]
        
        logits_torch = torch.from_numpy(logits)
        loss = loss_fn(logits_torch.view(-1, vocab_size), labels.view(-1))
        
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()
    
    avg_loss = total_loss / total_tokens
    
    print(f"Average Loss: {avg_loss}")
    
    prompts = ["In 2025", "I love data science because"]
    
    for prompt in prompts:
        print(generate_text(session, tokenizer, prompt, max_new_tokens=30))


if __name__ == "__main__":
    main()