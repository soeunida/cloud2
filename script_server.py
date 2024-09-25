import time
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from customed_pipeline import CustomedPipeline
from hf import NewPhi3Config
from model2 import CustomedPhi3ForCausalLM

# Set the random seed
torch.random.manual_seed(0)

# Load tokenizer and dataset
model_id = "microsoft/Phi-3-medium-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset("allenai/openbookqa", split='validation')
test = dataset.select(range(20))

# Initialize model and pipeline
config = NewPhi3Config(base_path='/nas/user/hayoung/')
model = CustomedPhi3ForCausalLM(config)
pipe = CustomedPipeline(model, config)

# Prefix for preprocessing
prefix = "\nRead the question and answer the following sentence in given multiple choice.\nAnswer only the sentence you chose. Never include a question and other word in your answer.\n\nquestion: "

# Preprocess the dataset
def preprocess(data):
    model_inputs = []
    for i in range(len(data['question_stem'])):
        offset = ord(data['answerKey'][i]) - ord('A')
        chat_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": prefix + data['question_stem'][i] + "\nchoices: ["
                }
            ],
            "answer": data['choices'][i]['text'][offset]
        }
        for j in range(4):
            chat_dict['messages'][0]['content'] += "\'" + data['choices'][i]['text'][j]
            if j < 3:
                chat_dict['messages'][0]['content'] += "\', "
            else:
                chat_dict['messages'][0]['content'] += "\']\n"
        model_inputs.append(chat_dict)
    return {"processed_data": model_inputs}

# Function to batchify data
def batchify(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Preprocess the test data
model_inputs = test.map(preprocess, batched=True, batch_size=5)
model_inputs = model_inputs['processed_data']
messages = [inputs['messages'] for inputs in model_inputs]
labels = [inputs['answer'] for inputs in model_inputs]

def run_inference(batch_size):
    batches = batchify(messages, batch_size)
    tokenized = [tokenizer.apply_chat_template(batch, 
                                               tokenize=True, 
                                               padding=True, 
                                               truncation=True,
                                               return_tensors="pt", 
                                               return_dict=True) for batch in batches]
    
    input_ids = [token['input_ids'].to('cuda') for token in tokenized]  # Move each tensor to GPU
    attention_mask = [token['attention_mask'].to('cuda') for token in tokenized]  # Move each tensor to GPU

    # Run inference and measure time
    start_time = time.time()
    
    # Pass input_ids and attention_mask for each batch
    all_generated_sentences = []
    for ids, mask in zip(input_ids, attention_mask):
        outputs = pipe.forward(ids, mask)  # Now pass tensor batches instead of list
        # Decode the generated outputs
        print(outputs)
        generated_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs['generated_sequence']]
        all_generated_sentences.extend(generated_sentences)
        

    inference_time = time.time() - start_time

    # Show generated sentences for this batch
    for idx, sentence in enumerate(all_generated_sentences):
        print(f"Generated sentence {idx + 1}: {sentence}")

    return inference_time, all_generated_sentences


# Save the inference times and generated sentences to a file
with open("batch_inference_results.txt", "w") as f:
    for batch_size in range(1, 75, 5):
        inference_time, generated_sentences = run_inference(batch_size)
        output_str = f"Batch Size: {batch_size}, Inference Time: {inference_time:.4f} seconds\n"
        
        # Print to console and write to file
        print(output_str.strip())
        f.write(output_str)
        
        # Save generated sentences to the file
        for idx, sentence in enumerate(generated_sentences):
            f.write(f"Generated sentence {idx + 1}: {sentence}\n")
