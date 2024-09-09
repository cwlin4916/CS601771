import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict

# 1. Load the SST2 dataset
dataset = load_dataset('stanfordnlp/sst2')

# 2. Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 3. Preprocessing function for tokenization
def preprocess(examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True)

# 4. Tokenize the dataset
tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["sentence"])

# 5. Split the training dataset into 95% for training and 5% for validation
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

# ---- FULL FINE-TUNING ---- #
# 8. Load the RoBERTa model for sequence classification
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# 9. Define the Trainer function
def get_trainer(model):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],  # Updated to use split training set
        eval_dataset=val_dataset,  # Use the 5% validation set for evaluation (correction here)
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

# 10. Initialize the trainer for full fine-tuning
trainer = get_trainer(model)

# 11. Train the model
trainer.train()

# 12. Save the full fine-tuned model and tokenizer for later use in evaluation
model.save_pretrained('./full_fine_tuned_roberta')
tokenizer.save_pretrained('./modified_roberta')
