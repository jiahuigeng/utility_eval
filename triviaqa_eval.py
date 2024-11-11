from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', load_in_4bit=True, device_map="auto")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the TriviaQA dataset
dataset = load_dataset('trivia_qa', 'unfiltered')

# Use select instead of slicing
eval_examples = dataset['validation'].select(range(40))

# Load evaluation metric
squad_metric = evaluate.load('squad')

# Lists to store predictions and references
predictions = []
references = []

# Evaluate
for example in eval_examples:
    question = example['question']
    # Prepare prompt
    prompt = f"Question: {question}\nAnswer:"

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)

    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer part from the output
    if 'Answer:' in generated_text:
        generated_answer = generated_text.split('Answer:')[-1].strip()
    else:
        generated_answer = generated_text
        
    generated_answer = generated_answer.split("Question")[0].strip()


    # Get gold answers
    references.append({
        'id': str(example['question_id']),
        'answers': {'text': [example['answer']['value']], 'answer_start': [0]}
    })
    predictions.append({
        'id': str(example['question_id']),
        'prediction_text': generated_answer
    })

# Compute final score
final_score = squad_metric.compute(predictions=predictions, references=references)
print(final_score)
