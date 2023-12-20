import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import json

 
model_name = "m3hrdadfi/zabanshenas-roberta-base-mix"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class MyDataset(Dataset):
    def __init__(self, tokenizer, filename, block_size=512):
        data_pairs = []
        with open(filename, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)    
       
        for dict_item in data_dict:
            print(dict_item)
            for key, value in dict_item.items():
                if(isinstance(value,dict)):
                    for key, val in value:
                        data_pairs.append(f"{key} : {' '.join(val)}")
                data_pairs.append(f"{key}: {' '.join(value)}")
        
        self.examples = [tokenizer(line, return_tensors="pt",padding="max_length", truncation=True, max_length=block_size)["input_ids"].squeeze() for line in data_pairs]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
    


dataset = MyDataset(tokenizer, 'dataset.json')
data_loader = DataLoader(dataset, batch_size=8)  # Adjust batch size according to your GPU memory

device = torch.device("mps")
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

model.save_pretrained("./robby")
tokenizer.save_pretrained("./robby")
