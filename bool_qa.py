import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate

# Load the Llama-2-7B model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load the BoolQ dataset
dataset = load_dataset("boolq", split="validation")

# Initialize the accuracy metric
accuracy_metric = evaluate.load("accuracy")

# List to store predictions and references
predictions = []
references = []

# Evaluate on a subset to avoid long runtime
sample_size = 100  # Adjust this to evaluate more or fewer samples
for i, example in enumerate(dataset):
    if i >= sample_size:
        break
    
    # Question and passage
    question = example["question"]
    passage = example["passage"]

    # Construct input prompt
    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer (true or false):"

    # Tokenize and encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Generate an answer
    with torch.no_grad():
        output = model.generate(inputs["input_ids"], max_new_tokens=5, num_beams=5, early_stopping=True)
    
    # Decode the generated answer
    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
    
    # Convert the generated answer to boolean
    if "true" in generated_answer:
        predicted_label = True
    elif "false" in generated_answer:
        predicted_label = False
    else:
        # If the model output is ambiguous, you could skip this example or set a default
        continue

    # Append the prediction and reference answer
    predictions.append(predicted_label)
    references.append(example["answer"])

# Calculate accuracy
accuracy = accuracy_metric.compute(predictions=predictions, references=references)
print(f"Accuracy on BoolQ: {accuracy['accuracy']:.4f}")
