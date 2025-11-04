from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import tensorflow as tf
import os

# 1️⃣ Load dataset — we’ll use IMDb for example (you can replace with your own)
dataset = load_dataset("imdb")

# 2️⃣ Load tokenizer and tokenize dataset
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3️⃣ Prepare TensorFlow datasets
train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size=8
)
test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=8
)

# 4️⃣ Load DistilBERT model
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 5️⃣ Compile and train model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(train_dataset, validation_data=test_dataset, epochs=1)  # You can increase epochs

# 6️⃣ Save fine-tuned model
MODEL_DIR = r"E:\Sentiment_Analyesis_project_01\models\distilbert_finetuned"
os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("✅ Fine-tuned model saved successfully at:", MODEL_DIR)
