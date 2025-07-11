import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizer

class DatasetTransformersLike(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, device=None, max_length=512):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def __len__(self):
        return len(self.texts)

    def tokenize_function(self, samples):
        return self.tokenizer(
            samples,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized = self.tokenize_function(text)

        return {
            "input_ids": tokenized["input_ids"].squeeze(0).to(self.device),
            "attention_mask": tokenized["attention_mask"].squeeze(0).to(self.device),
            "labels": label.to(self.device)
        }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, device=torch.device("mps" if torch.backends.mps.is_available() else "cpu" ), max_length=512):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def tokenize_function(self, samples):
        return self.tokenizer(samples, padding="max_length", truncation=True, max_length=self.max_length)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_tokenized = self.tokenize_function(text)
        mask = torch.tensor(text_tokenized["attention_mask"], dtype=torch.bool, device=self.device)
        input_ids = torch.tensor(text_tokenized["input_ids"], dtype=torch.long, device=self.device)
        return input_ids, label, mask

class Data:
    def __init__(self, bert_token = False, train_file = "../data/train_dataset.csv", test_file = "../data/test_dataset.csv", val_file = "../data/validation_dataset.csv", max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if bert_token else AutoTokenizer.from_pretrained("bert-base-uncased")
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.output_dim = None
        self.max_length = max_length

    def load_data(self, batch_size, device=torch.device("mps" if torch.backends.mps.is_available() else "cpu" )):
        df_train = pd.read_csv(self.train_file)
        df_val = pd.read_csv(self.val_file)
        df_test = pd.read_csv(self.test_file)
        self.output_dim = len(np.unique(df_train['label'].values))
        x_train, y_train = df_train['text'], df_train['label']
        x_val, y_val = df_val['text'], df_val['label']
        x_test, y_test = df_test['text'], df_test['label']

        train_dataset = Dataset(texts=x_train, labels=y_train, device=device, tokenizer=self.tokenizer, max_length=self.max_length)
        val_dataset = Dataset(texts=x_val, labels=y_val, device=device, tokenizer=self.tokenizer, max_length=self.max_length)
        test_dataset = Dataset(texts=x_test, labels=y_test, device=device, tokenizer=self.tokenizer, max_length=self.max_length)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def load_data_transformers_like(self, device=torch.device("mps" if torch.backends.mps.is_available() else "cpu" )):
        df_train = pd.read_csv(self.train_file)
        df_val = pd.read_csv(self.val_file)
        df_test = pd.read_csv(self.test_file)
        self.output_dim = len(np.unique(df_train['label'].values))
        x_train, y_train = df_train['text'], df_train['label']
        x_val, y_val = df_val['text'], df_val['label']
        x_test, y_test = df_test['text'], df_test['label']

        train_dataset = DatasetTransformersLike(texts=x_train, labels=y_train, device=device, tokenizer=self.tokenizer, max_length=self.max_length)
        val_dataset = DatasetTransformersLike(texts=x_val, labels=y_val, device=device, tokenizer=self.tokenizer, max_length=self.max_length)
        test_dataset = DatasetTransformersLike(texts=x_test, labels=y_test, device=device, tokenizer=self.tokenizer, max_length=self.max_length)
        return train_dataset, val_dataset, test_dataset

    def create_dataset(self):
        ds = load_dataset("dair-ai/emotion", "split")
        pd.DataFrame(ds["train"]).to_csv(self.train_file, index=False)
        pd.DataFrame(ds["test"]).to_csv(self.test_file, index=False)
        pd.DataFrame(ds["validation"]).to_csv(self.val_file, index=False)
