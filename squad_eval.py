from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import evaluate
import torch
from tqdm import tqdm

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', load_in_4bit=True, device_map="auto")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the SQuAD dataset
dataset = load_dataset('squad')

# Evaluate on the first 200 examples from the validation set
eval_examples = dataset['validation'].select(range(200))

# Load evaluation metric
squad_metric = evaluate.load('squad')

# Lists to store predictions and references
predictions = []
references = []

# Evaluate
for example in tqdm(eval_examples):
    context = example['context']
    question = example['question']
    reference_answers = example['answers']['text']  # List of acceptable answers

    # Prepare prompt
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Encode the prompt
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=2048,  # Adjust max_length to fit within model's context window
    ).to(device)

    # Generate answer
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=50,  # Maximum number of tokens to generate
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer from the generated text
    if 'Answer:' in generated_text:
        generated_answer = generated_text.split('Answer:')[-1].strip()
    else:
        generated_answer = generated_text

    # Append prediction and reference for evaluation
    predictions.append({
        'id': example['id'],
        'prediction_text': generated_answer,
    })
    references.append({
        'id': example['id'],
        'answers': {
            'text': reference_answers,
            'answer_start': example['answers']['answer_start'],
        }
    })

# Compute final scores
results = squad_metric.compute(predictions=predictions, references=references)
print(results)
