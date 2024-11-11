import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
from rouge_score import rouge_scorer

# Load the Llama-2-7B model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
model.eval()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load the XSum dataset
dataset = load_dataset("xsum", split="test")

# Initialize ROUGE scorer
rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Initialize variables to accumulate scores
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

# Evaluate on a subset of the dataset to avoid long runtime
sample_size = 100  # Adjust this number to control the evaluation subset size
for i, example in enumerate(dataset):
    if i >= sample_size:
        break
    
    # Article text (source for summarization)
    article_text = example["document"]
    reference_summary = example["summary"]
    
    prompt = f"summarize the following text into one sentence: {article_text}"
    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Generate summary
    with torch.no_grad():
        output = model.generate(inputs["input_ids"], max_new_tokens=512, num_beams=5, early_stopping=True)
    
    # Decode generated summary
    generated_summary = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Prompt:", prompt)
    print("Reference Summary:", reference_summary)
    print("Generated Summary:", generated_summary)
    # Compute ROUGE scores for the generated summary
    scores = rouge_scorer.score(generated_summary, reference_summary)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

# Calculate average ROUGE scores
average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

# Display results
print(f"Average ROUGE-1 Score: {average_rouge1:.4f}")
print(f"Average ROUGE-2 Score: {average_rouge2:.4f}")
print(f"Average ROUGE-L Score: {average_rougeL:.4f}")
