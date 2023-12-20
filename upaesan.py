import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./robby"  # Path where you saved your fine-tuned model
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True,is_decoder=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

def generate_text(input_text, max_length=150):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate text using the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length)

    # Decode and return the output text
    return tokenizer.decode(output[0], skip_special_tokens=True)

input_text = "a menna' "
generated_text = generate_text(input_text)
print(generated_text)