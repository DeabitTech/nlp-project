import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import json

 
model_name = "m3hrdadfi/zabanshenas-roberta-base-mix"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class MyDataset(Dataset):
    def __init__(self, tokenizer, filename, block_size=512):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = json.load(f)
        self.examples = [tokenizer(line, return_tensors="pt", truncation=True, max_length=block_size)["input_ids"].squeeze() for line in lines]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
    


dataset = MyDataset(tokenizer, 'dataset.json')
data_loader = DataLoader(dataset, batch_size=8)  # Adjust batch size according to your GPU memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):  # Number of epochs
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

model.save_pretrained("./")
