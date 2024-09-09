import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from peft import PeftModel

# 1. Load the SST2 dataset
dataset = load_dataset('stanfordnlp/sst2')
print("Original Dataset Structure:", dataset)

# Define a preprocessing function for BERT and RoBERTa separately
def preprocess(tokenizer, examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True)

# 3. Define the evaluation function
def evaluate_sample_model(model: torch.nn.Module, dataset: torch.utils.data.Dataset, data_collator) -> Dict[str, float]:
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)  # Use the collator for padding
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    metric = evaluate.load('accuracy')  # Load accuracy metric

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=-1)

        metric.add_batch(predictions=predicted_labels, references=batch['labels'])

    return metric.compute()  # Compute accuracy

# ---- FULL FINE-TUNED ROBERTA MODEL EVALUATION ---- #
# 4. Load the RoBERTa tokenizer and model (for full fine-tuned RoBERTa)
roberta_tokenizer_full = RobertaTokenizer.from_pretrained('./modified_roberta')

# Tokenize the datasets for RoBERTa-based full fine-tuned model
train_val_split = dataset['train'].train_test_split(test_size=0.05, seed=42)
validation_dataset_full = train_val_split['test'].map(lambda x: preprocess(roberta_tokenizer_full, x), batched=True, remove_columns=["sentence"])
validation_dataset_full = validation_dataset_full.rename_column('label', 'labels')
validation_dataset_full.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

test_dataset_full = dataset['validation'].map(lambda x: preprocess(roberta_tokenizer_full, x), batched=True, remove_columns=["sentence"])
test_dataset_full = test_dataset_full.rename_column('label', 'labels')
test_dataset_full.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize the data collator for RoBERTa full fine-tuned model
roberta_data_collator_full = DataCollatorWithPadding(tokenizer=roberta_tokenizer_full)

# 5. Load and evaluate the full fine-tuned RoBERTa model
full_model = RobertaForSequenceClassification.from_pretrained('./full_fine_tuned_roberta')

print("\n--- Full Fine-Tuned RoBERTa Model Evaluation on Validation (5% Split) ---")
full_val_performance = evaluate_sample_model(full_model, validation_dataset_full, roberta_data_collator_full)
print("Performance (Full Fine-Tuned RoBERTa Model on Validation):", full_val_performance)

print("\n--- Full Fine-Tuned RoBERTa Model Evaluation on Test Set (SST-2 Validation) ---")
full_test_performance = evaluate_sample_model(full_model, test_dataset_full, roberta_data_collator_full)
print("Performance (Full Fine-Tuned RoBERTa Model on Test Set):", full_test_performance)

# ---- LoRA FINE-TUNED ROBERTA MODEL EVALUATION ---- #
# Load the RoBERTa tokenizer and LoRA model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
lora_model_base = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
lora_model = PeftModel.from_pretrained(lora_model_base, './lora_fine_tuned_roberta')

# Tokenize the datasets for RoBERTa-based models (LoRA fine-tuned)
validation_dataset_roberta = train_val_split['test'].map(lambda x: preprocess(roberta_tokenizer, x), batched=True, remove_columns=["sentence"])
validation_dataset_roberta = validation_dataset_roberta.rename_column('label', 'labels')
validation_dataset_roberta.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

test_dataset_roberta = dataset['validation'].map(lambda x: preprocess(roberta_tokenizer, x), batched=True, remove_columns=["sentence"])
test_dataset_roberta = test_dataset_roberta.rename_column('label', 'labels')
test_dataset_roberta.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize the data collator for RoBERTa LoRA fine-tuned model
roberta_data_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)

print("\n--- LoRA Fine-Tuned RoBERTa Model Evaluation on Validation (5% Split) ---")
lora_val_performance = evaluate_sample_model(lora_model, validation_dataset_roberta, roberta_data_collator)
print("Performance (LoRA Fine-Tuned RoBERTa Model on Validation):", lora_val_performance)

print("\n--- LoRA Fine-Tuned RoBERTa Model Evaluation on Test Set (SST-2 Validation) ---")
lora_test_performance = evaluate_sample_model(lora_model, test_dataset_roberta, roberta_data_collator)
print("Performance (LoRA Fine-Tuned RoBERTa Model on Test Set):", lora_test_performance)

# ---- BITFIT FINE-TUNED ROBERTA MODEL EVALUATION ---- #
# Load and evaluate the BitFit fine-tuned model (uses RoBERTa tokenizer and model)
roberta_tokenizer_bitfit = RobertaTokenizer.from_pretrained('./modified_roberta_bitfit')

# Tokenize the datasets for RoBERTa-based models (BitFit fine-tuned)
validation_dataset_bitfit = train_val_split['test'].map(lambda x: preprocess(roberta_tokenizer_bitfit, x), batched=True, remove_columns=["sentence"])
validation_dataset_bitfit = validation_dataset_bitfit.rename_column('label', 'labels')
validation_dataset_bitfit.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

test_dataset_bitfit = dataset['validation'].map(lambda x: preprocess(roberta_tokenizer_bitfit, x), batched=True, remove_columns=["sentence"])
test_dataset_bitfit = test_dataset_bitfit.rename_column('label', 'labels')
test_dataset_bitfit.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize the data collator for RoBERTa BitFit fine-tuned model
roberta_data_collator_bitfit = DataCollatorWithPadding(tokenizer=roberta_tokenizer_bitfit)

# Load and evaluate the BitFit fine-tuned RoBERTa model
bitfit_model = RobertaForSequenceClassification.from_pretrained('./modified_roberta_bitfit')

print("\n--- BitFit Fine-Tuned RoBERTa Model Evaluation on Validation (5% Split) ---")
bitfit_val_performance = evaluate_sample_model(bitfit_model, validation_dataset_bitfit, roberta_data_collator_bitfit)
print("Performance (BitFit Fine-Tuned RoBERTa Model on Validation):", bitfit_val_performance)

print("\n--- BitFit Fine-Tuned RoBERTa Model Evaluation on Test Set (SST-2 Validation) ---")
bitfit_test_performance = evaluate_sample_model(bitfit_model, test_dataset_bitfit, roberta_data_collator_bitfit)
print("Performance (BitFit Fine-Tuned RoBERTa Model on Test Set):", bitfit_test_performance)
