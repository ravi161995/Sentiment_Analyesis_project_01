import streamlit as st
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
import os

# ✅ 1️⃣ Load fine-tuned model (correct folder)
MODEL_DIR = r"E:\Sentiment_Analyesis_project_01\models\distilbert_finetuned"

if not os.path.exists(MODEL_DIR):
    st.error(f"Model folder not found at {MODEL_DIR}")
else:
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    # ✅ 2️⃣ Streamlit UI
    st.title("💬 Sentiment Analysis App (Fine-tuned DistilBERT)")
    st.write("This app predicts whether a sentence expresses **Positive** or **Negative** sentiment.")

    # ✅ 3️⃣ User input
    text = st.text_area("Enter your text here:", "I love this product!")

    if st.button("Predict Sentiment"):
        if text.strip() == "":
            st.warning("Please enter some text first.")
        else:
            # ✅ 4️⃣ Tokenize and predict correctly
            inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = int(tf.argmax(logits, axis=1).numpy()[0])

            # ✅ 5️⃣ Display result
            label = "😊 Positive" if predicted_class == 1 else "😠 Negative"
            st.subheader(f"Prediction: {label}")
