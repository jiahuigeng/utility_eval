import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# Load the Llama-2-7B model and tokenizer
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load the CommonsenseQA dataset
dataset = load_dataset("commonsense_qa", split="validation")

# Initialize variable to accumulate accuracy
correct_predictions = 0
total_samples = 100  # Adjust the sample size to control the evaluation subset size

# Define answer choice mapping
answer_choices = ["A", "B", "C", "D", "E"]

def get_filtered_answer(resp):
    for char in resp:
        if char.isupper():
            return char
    return None
        

# Loop through a subset of the dataset for evaluation
for example in tqdm(dataset.select(range(total_samples))):
    question = example["question"]
    choices = example["choices"]["text"]
    
    # Find the index of the correct answer
    correct_answer_key = example["answerKey"]
    correct_answer_index = answer_choices.index(correct_answer_key)
    correct_answer = choices[correct_answer_index]

    # Construct a prompt with the question and choices
    prompt = f"Question: {question}\n"
    for idx, choice in enumerate(choices):
        prompt += f"{answer_choices[idx]}. {choice}\n"
    prompt += "Answer:"

    # Tokenize and encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Generate an answer
    with torch.no_grad():
        output = model.generate(inputs["input_ids"], max_new_tokens=5, num_beams=1, early_stopping=True)

    # Decode the model's answer
    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    predicted_choice = generated_answer.split("Answer:")[-1].strip()
    predicted_choice = get_filtered_answer(predicted_choice)
    print(f"predicted_choice: {predicted_choice}; correct_answer_key: {correct_answer_key}")
    # Check if the predicted choice matches the correct answer key
    if predicted_choice == correct_answer_key:
        correct_predictions += 1

# Calculate and print accuracy
accuracy = correct_predictions / total_samples
print(f"Accuracy on CommonsenseQA: {accuracy:.4f}")
