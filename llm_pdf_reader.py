import time
import traceback
import PyPDF2
import pandas as pd
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        if not text.strip():
            print("⚠️ No text extracted! Please check the PDF file.")
        else:
            print("✅ Text extracted successfully!")
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
    return text

# List of models to compare
models = {
    "Summarization": [
        "facebook/bart-large-cnn", "t5-small", "t5-base", "t5-large", 
        "google/pegasus-xsum", "flan-t5-base"
    ],
    "Question-Answering": [
        "deepset/roberta-base-squad2", "bert-large-uncased-whole-word-masking-finetuned-squad",
        "distilbert-base-cased-distilled-squad", "microsoft/deberta-v3-large"
    ]
}

# Define max input length per model
model_max_length = {
    "facebook/bart-large-cnn": 1024,
    "t5-small": 512,
    "t5-base": 512,
    "t5-large": 512,
    "google/pegasus-xsum": 1024,
    "flan-t5-base": 512
}

def evaluate_models(text, question):
    """Compare different models for summarization and question-answering."""
    results = []
    
    for model_type, model_list in models.items():
        for model_name in model_list:
            try:
                print(f"🚀 Evaluating {model_name} for {model_type}...")
                start_time = time.time()
                
                # Load the appropriate model pipeline
                pipe = pipeline("summarization" if model_type == "Summarization" else "question-answering", model=model_name)
                
                # Dynamically adjust input size based on model limit
                max_length = model_max_length.get(model_name, 512)
                truncated_text = text[:max_length]
                print(f"⚠️ Truncating input text to {max_length} tokens for {model_name}...")

                if model_type == "Summarization":
                    output = pipe(truncated_text, max_length=150, min_length=50, do_sample=False)
                    result_text = output[0]['summary_text']
                else:
                    output = pipe(question=question, context=truncated_text)
                    result_text = output['answer']
                
                end_time = time.time()
                
                print(f"✅ Model {model_name} output (first 100 chars): {result_text[:100]}...")
                
                results.append({
                    "Model": model_name,
                    "Type": model_type,
                    "Output": result_text,
                    "Time Taken (s)": round(end_time - start_time, 2)
                })
            except Exception as e:
                print(f"❌ Error with model {model_name}: {traceback.format_exc()}")
                results.append({
                    "Model": model_name,
                    "Type": model_type,
                    "Output": f"Error: {e}",
                    "Time Taken (s)": "N/A"
                })

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    pdf_path = "example.pdf"  # Update with your actual PDF file path

    print("📂 Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if not extracted_text.strip():
        print("⚠️ No text extracted! Exiting program.")
    else:
        question = "What is the main conclusion of the paper?"

        print("🔍 Running model evaluation...")
        comparison_table = evaluate_models(extracted_text, question)

        if comparison_table.empty:
            print("⚠️ No results generated! Check model execution.")
        else:
            print("✅ Saving CSV file now...")
            print(comparison_table)  # Print table before saving
            comparison_table.to_csv("model_comparison_results.csv", index=False)
            print("✅ File saved successfully!")

        print("\n📊 Final Comparison Table:\n")
        print(comparison_table.to_string())