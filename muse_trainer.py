from datasets import load_dataset, Image, Dataset
from muse_maskgit_pytorch import VQGanVAETaming
from pathlib import Path
import random
from tqdm import tqdm
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
from PIL import Image
from transformers import T5Tokenizer
import requests
import types
from joblib import Parallel, delayed
from multiprocessing import cpu_count

laion_dataset = load_dataset("laion/laion2B-en-aesthetic", streaming=True)
laion_dataset = laion_dataset.with_format("torch")

valid_dataset = laion_dataset["train"].take(1000)

name = "google/t5-v1_1-base"
tokenizer = T5Tokenizer.from_pretrained(name)

def convert(x):
    return x.convert('RGB')

def download_image(data):
    try:
        image = Image.open(requests.get(data['URL'], stream=True, timeout=5).raw)
        transform_list = [
            T.Lambda(convert),
            T.Resize(256),
            T.RandomHorizontalFlip(),
            T.CenterCrop(256),
            T.ToTensor()
        ]
        transform = T.Compose(transform_list)
        image = transform(image)
        return image
    except:
        return None

def map_to_text(example):
    return example['TEXT']

def collate_fn(examples):
    encoded = tokenizer.batch_encode_plus(
        list(map(map_to_text, examples)),
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    )
    input_ids = encoded['input_ids'].tolist()
    attention_mask = encoded['attention_mask'].tolist()
    
    images = Parallel(n_jobs=cpu_count(), prefer="threads")(delayed(download_image)(example) for i, example in enumerate(examples))
    indexes_to_remove = []
    for i, image in enumerate(images):
        if image is None:
            indexes_to_remove.append(i)

    images = [img for img in images if img is not None]
    input_ids = [i for j, i in enumerate(input_ids) if j not in indexes_to_remove]
    attention_mask = [i for j, i in enumerate(attention_mask) if j not in indexes_to_remove]

    images = torch.stack(images)
    return images, torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask)

def get_dataloaders(batch_size):
    dataloader = DataLoader(laion_dataset["train"], batch_size=batch_size, collate_fn=collate_fn, num_workers=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2)
    return dataloader, valid_dataloader

if __name__ == '__main__':
    dataloader, valid_dataloader = get_dataloaders()
    for i, batch in enumerate(tqdm(valid_dataloader, total=1000)):
        print(batch['input_ids'].shape, batch['images'].shape, batch['attention_mask'].shape)