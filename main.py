from transformer_model import Transformer
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from exporting import export_to_onnx

def main():
    dataset_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "train")
    train = dataset_train.shuffle(seed=42).select(range(1000))

    dataset_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "test")
    test = dataset_test.shuffle(seed=42).select(range(500))

    dataset_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "validation")
    val = dataset_val.shuffle(seed=42).select(range(500))

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    def tokenize_data_labels(ex):
        curr_ex = tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
        print("HERE",len(curr_ex[0]))
        print(type(curr_ex))
        labels = []
        for input_id_row in curr_ex["input_ids"]:
            # print("inside", input_id_row)
            shifted = input_id_row[1:] + [pad_token_id]
            processed_row = [-100 if token_id == pad_token_id else token_id for token_id in shifted] #if there is padding token, then replace it with -100
            labels.append(processed_row)
        curr_ex["labels"] = labels
        print(len(curr_ex[0]))
        return curr_ex
    
    train_tok = train.map(tokenize_data_labels, batched=True, remove_columns=["text"])
    val_tok = val.map(tokenize_data_labels, batched=True, remove_columns=["text"])
    test_tok = test.map(tokenize_data_labels, batched=True, remove_columns=["text"])

    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataloader = DataLoader(train_tok, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_tok, batch_size=8)

    vocab_size = tokenizer.vocab_size

    batch = next(iter(train_dataloader))
    # print(batch)
    # print("shape of one batch:")
    # print(batch['input_ids'].shape)
    # print("example input_ids:")
    # print(batch['input_ids'][0])
    # print("example labels:")
    # print(batch['labels'][0])
    #for batch in Dataloader(train_tok, batch_size=16):
    hidden_dim = 768
    d_model = 512
    n_blocks = 3
    max_seq_len = 128
    model = Transformer(hidden_dim, d_model, max_seq_len, vocab_size, n_blocks)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    n_epochs = 5

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            X = batch["input_ids"]
            y = batch["labels"]

            optimizer.zero_grad()

            logits = model(X, batch["attention_mask"])

            loss = loss_fn(logits.view(-1,vocab_size), y.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"For Epoch {epoch}, average training_loss is {avg_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                X = batch["input_ids"]
                y = batch["labels"]
                attention_mask = batch["attention_mask"]
                
                logits = model(X, attention_mask)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"For Epoch {epoch}, average validation_loss is {avg_val_loss}")

    onnx_path = "transformer_model.onnx"
    export_to_onnx(model, vocab_size, max_seq_len, onnx_path)

if __name__=="__main__":
    main()

    