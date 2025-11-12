import torch
import torch.onnx

def export_to_onnx(model, vocab_size, max_seq_len=128, save_path='transformer_model.onnx'):
    model.eval()
    
    batch_size = 1
    seq_len = max_seq_len
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    
    torch.onnx.export(
        model,                          
        (dummy_input_ids, dummy_attention_mask),  
        save_path,                      
        input_names=['input_ids', 'attention_mask'],    
        output_names=['logits'],                        
        dynamic_axes={                  
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'}
        },
        do_constant_folding=True
    )
    
    print(f"Model exported successfully to {save_path}")