from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import evaluate
import torch
from tqdm import tqdm

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', load_in_4bit=True, device_map="auto")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the CNN/DailyMail dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')

# Evaluate on the first 200 examples from the test split
eval_examples = dataset['test'].select(range(200))

# Load evaluation metric
rouge = evaluate.load('rouge')

# Lists to store predictions and references
predictions = []
references = []

# Evaluate
for example in tqdm(eval_examples):
    article = example['article']
    reference_summary = example['highlights']

    # Prepare prompt
    prompt = f"Summarize the following article:\n{article}\nSummary:"

    # Encode the prompt
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=2048,  # Adjust to ensure total length <= 4096
        padding='max_length',
    ).to(device)

    # Generate summary
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=256,  # Number of tokens to generate
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the summary from the generated text
    if 'Summary:' in generated_text:
        generated_summary = generated_text.split('Summary:')[-1].strip()
    else:
        generated_summary = generated_text

    predictions.append(generated_summary)
    references.append(reference_summary)

# Compute ROUGE scores
results = rouge.compute(predictions=predictions, references=references)
print(results)
