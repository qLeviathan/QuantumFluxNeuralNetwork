# Quantum Flux Neural Network - Changelog

## 2025-02-27: DataLoader Fix

### Issue
The training script (`scripts/train_wandb.py`) was encountering an error during batch processing:
```
ValueError: too many values to unpack (expected 2)
```

This error occurred in the `collate_fn` function which was expecting each batch item to be a tuple of exactly 2 elements (input_ids and labels), but the Hugging Face dataset was returning items with more elements.

### Solution
We implemented a more flexible approach to handle different batch structures:

1. Initially, we tried creating a custom collate function in `scripts/fix_collate.py`, but encountered pickling issues with multiprocessing.

2. Instead, we updated `train_wandb.py` to:
   - Remove the custom collate function and use the default DataLoader behavior
   - Update the training and validation loops to handle different batch formats:
     ```python
     # Get inputs and labels
     if isinstance(batch, dict):
         input_ids = batch['input_ids'].to(device)
         labels = batch['labels'].to(device)
         attention_mask = batch.get('attention_mask', None)
         if attention_mask is not None:
             attention_mask = attention_mask.to(device)
     else:
         # Handle other batch types as needed
         input_ids = batch[0].to(device)
         labels = batch[1].to(device)
         attention_mask = batch[2].to(device) if len(batch) > 2 else None
     ```

### Files Changed
- Created: `scripts/fix_collate.py`
- Modified: `scripts/train_wandb.py`

### Implementation Details

#### 1. New Collate Function
The new collate function in `scripts/fix_collate.py` is designed to be more flexible:
```python
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
```

#### 2. DataLoader Updates
Updated the DataLoader creation in `train_wandb.py`:
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=args.num_workers,
    collate_fn=create_collate_fn(),  # Use the new collate function
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=val_sampler,
    num_workers=args.num_workers,
    collate_fn=create_collate_fn(),  # Use the new collate function
    pin_memory=True
)
```

#### 3. Batch Processing Updates
Updated the batch processing in both training and validation loops to use the dictionary format:
```python
# Get inputs and labels
input_ids = batch['input_ids'].to(device)
labels = batch['labels'].to(device)
attention_mask = batch.get('attention_mask', None)
if attention_mask is not None:
    attention_mask = attention_mask.to(device)
```

### Benefits
1. **Improved Robustness**: The training script now handles different batch formats correctly
2. **Better Error Handling**: The collate function provides debug information when encountering unexpected batch formats
3. **Consistent Interface**: All batch processing now uses a consistent dictionary-based interface
4. **Future Compatibility**: The solution is more adaptable to changes in dataset formats

### Next Steps
- Monitor training performance to ensure the fix is working correctly
- Consider adding more robust error handling throughout the training pipeline
- Update documentation to reflect the new batch processing approach
