import torch

def create_collate_fn():
    def collate_fn(batch):
        # Handle different batch structures
        if isinstance(batch[0], dict):
            # For dictionary items from Hugging Face datasets
            input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
            labels = torch.stack([torch.tensor(item['labels']) for item in batch])
            attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch]) if 'attention_mask' in batch[0] else None
        else:
            # For tuple items, print format and extract appropriately
            print(f"DEBUG - Batch item type: {type(batch[0])}, Length: {len(batch[0]) if isinstance(batch[0], (list, tuple)) else 'N/A'}")
            
            # Take first element as input_ids and second as labels
            input_ids = torch.stack([torch.tensor(item[0]) for item in batch])
            labels = torch.stack([torch.tensor(item[1]) for item in batch])
            attention_mask = torch.stack([torch.tensor(item[2]) for item in batch]) if len(batch[0]) > 2 else None
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    return collate_fn
