import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
from copy import deepcopy
from typing import OrderedDict, Union, List, Dict  # <-- Add this import

# --- Add the helper functions here ---
def freeze_parameters(hf_model: torch.nn.Module, learnable_biases: Union[str, List[str]] = 'all'):
    """
    Freezes all parameters except for bias terms.
    """
    learnable_biases = ['bias'] if learnable_biases == 'all' else learnable_biases

    if not isinstance(learnable_biases, list):
        learnable_biases = [learnable_biases]

    for par_name, par_tensor in hf_model.base_model.named_parameters():
        par_tensor.requires_grad = any(
            ('bias' in par_name and kw in par_name) for kw in learnable_biases
        )

    return hf_model

def get_trainable_parameters(hf_model: torch.nn.Module):
    """
    Gets the parameters that are being trained.
    """
    return {
        par_name: par_tensor
        for par_name, par_tensor in hf_model.named_parameters()
        if par_tensor.requires_grad
    }

def get_offsets(base_model: Union[torch.nn.Module, OrderedDict, Dict], finetuned_model: Union[torch.nn.Module, OrderedDict, Dict]):
    """
    Computes the difference (offsets) between the original and fine-tuned models.
    """
    if isinstance(finetuned_model, torch.nn.Module):
        finetuned_model = get_trainable_parameters(finetuned_model)

    if isinstance(base_model, torch.nn.Module):
        base_model = base_model.state_dict()

    return {
        'offsets': {
            param_name: param_tensor - base_model[param_name]
            for param_name, param_tensor in finetuned_model.items()
            if 'classifier' not in param_name  # Apply only to non-classifier layers
        },
        'classifier': {
            param_name: param_tensor
            for param_name, param_tensor in finetuned_model.items()
            if 'classifier' in param_name  # Keep classifier weights intact
        }
    }

def save_bitfit(base_model: Union[torch.nn.Module, OrderedDict, Dict], finetuned_model: Union[torch.nn.Module, OrderedDict, Dict], path: str):
    """
    Saves the BitFit offsets (bias changes) to a file.
    """
    torch.save(get_offsets(base_model, finetuned_model), path)

# --- End of helper functions ---

# 1. Check if MPS or GPU is available, otherwise fall back to CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load the SST2 dataset
dataset = load_dataset('stanfordnlp/sst2')

# 3. Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 4. Preprocess the dataset using RoBERTa tokenizer
def preprocess(examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["sentence"])

# 5. Split the training dataset into 90% training and 10% validation
train_test_split_dataset = tokenized_dataset['train'].train_test_split(test_size=0.05, seed=42)
train_data = train_test_split_dataset['train']
val_data = train_test_split_dataset['test']

# Convert split datasets back into DatasetDict format
train_dataset = DatasetDict({"train": train_data})
val_dataset = DatasetDict({"validation": val_data})

# Use the SST-2 validation set as the test set
test_dataset = tokenized_dataset['validation']

# 6. Load the base RoBERTa model for sequence classification
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# 7. Move both the original model and the fine-tuned model to the same device
model = model.to(device)
original_model = deepcopy(model).to(device)  # Copy and move the original model

# 8. Freeze all model parameters except for the bias vectors using BitFit
freeze_parameters(model, learnable_biases=['bias'])  # Explicitly freeze non-bias parameters

# 9. Set up training arguments (adapted for BitFit)
training_args = TrainingArguments(
    output_dir='./results_bitfit',
    evaluation_strategy='steps',
    save_steps=500,
    learning_rate=1e-4,  # A higher learning rate may be needed for bias-only fine-tuning
    per_device_train_batch_size=16,
    num_train_epochs=3,  # Bias terms may converge faster; you can experiment with this
    weight_decay=0.01,
    logging_dir='./logs_bitfit',
)

# 10. Initialize the trainer with the BitFit model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset['train'],  # Updated to use split training set
    eval_dataset=val_dataset['validation'],  # Updated to use 10% validation set
    tokenizer=tokenizer,  # Use RoBERTa tokenizer
)
# 11. Train the BitFit model
trainer.train()

# 12. Save the full fine-tuned model (including config and weights)
model.save_pretrained('./modified_roberta_bitfit')

# 13. Save the tokenizer (in case you want to reuse it with this fine-tuned model)
tokenizer.save_pretrained('./modified_roberta_bitfit')

# 14. (Optional) Save the bias offsets using the BitFit helper function (for analysis)
save_bitfit(original_model, model, 'sentiment_analysis_bitfit.pt')
