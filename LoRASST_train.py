import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model

# 1. Load the SST2 dataset
dataset = load_dataset('stanfordnlp/sst2')

# 2. Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 3. Preprocessing function for tokenization
def preprocess(examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True)

# 4. Tokenize the dataset
tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["sentence"])

# 5. Split the training dataset into 95% training and 5% validation
train_test_split_dataset = tokenized_dataset['train'].train_test_split(test_size=0.05, seed=42)
train_data = train_test_split_dataset['train']
val_data = train_test_split_dataset['test']

# Convert split datasets back into DatasetDict format
train_dataset = DatasetDict({"train": train_data})
val_dataset = DatasetDict({"validation": val_data})

# Use the SST-2 validation set as the test set
test_dataset = tokenized_dataset['validation']

# 6. Initialize the data collator (for padding)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 7. Define training arguments
training_args = TrainingArguments(
    output_dir='./results',                 
    evaluation_strategy='steps',            # Evaluate every few steps
    save_steps=500,                         # Save the model every 500 steps
    learning_rate=2e-5,                     # Learning rate
    per_device_train_batch_size=16,         # Batch size per device
    num_train_epochs=3,                     # Number of epochs
    weight_decay=0.01,                      # Weight decay to prevent overfitting
    logging_dir='./logs',                   # Directory for logging
)

# 8. Define the Trainer function
def get_trainer(model):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],  # Updated to use split training set
        eval_dataset=val_dataset,  # Use the 5% validation set for evaluation (corrected)
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

# ---- LoRA FINE-TUNING ---- #
# 9. Re-initialize a fresh RoBERTa model for LoRA fine-tuning
lora_model_base = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# 10. Initialize LoRA configuration
peft_config = LoraConfig(
    task_type="SEQ_CLS",                     
    inference_mode=False,                    
    r=8,                                     
    lora_alpha=16,                           
    lora_dropout=0.1,                        
)

# 11. Apply LoRA to the fresh RoBERTa model
lora_model = get_peft_model(lora_model_base, peft_config)

# 12. Initialize the trainer for LoRA fine-tuning
lora_trainer = get_trainer(lora_model)

# 13. Train the LoRA fine-tuned model
lora_trainer.train()

# 14. Save the LoRA fine-tuned model for evaluation
lora_model.save_pretrained('./lora_fine_tuned_roberta')

# 15. Save the tokenizer for LoRA model (in case it’s needed later)
tokenizer.save_pretrained('./lora_fine_tuned_roberta')
