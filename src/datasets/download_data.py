from torch.utils.data import DataLoader
from transformers import MarianTokenizer
from datasets import load_dataset
import torch

class TranslationDataModule:
    def __init__(
        self,
        batch_size=4,
        max_length=64,
        train_size=10000,
        eval_size=1000,
        test_size=1000,
        num_workers=4,
        pin_memory=True,
        model_name="Helsinki-NLP/opus-mt-en-ru"
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_size = train_size
        self.eval_size = eval_size
        self.test_size = test_size
        self.model_name = model_name
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.src_vocab_size = self.tokenizer.vocab_size
        self.tgt_vocab_size = self.tokenizer.vocab_size

        self.dataset = None
        self.tokenized_train = None
        self.tokenized_eval = None
        self.tokenized_test = None

    def preprocess(self, example):
        """
        Предобработка одного примера данных
        """
        src = example["translation"]["en"]
        tgt = example["translation"]["ru"]
        
        src_enc = self.tokenizer(
            src, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length
        )
        tgt_enc = self.tokenizer(
            tgt, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length
        )
        
        return {
            "input_ids": src_enc["input_ids"],
            "attention_mask": src_enc["attention_mask"],
            "labels": tgt_enc["input_ids"]
        }

    def collate_fn(self, batch):
        """
        Преобразование батча списков в тензоры
        """
        input_ids = torch.tensor([item["input_ids"] for item in batch])
        attention_mask = torch.tensor([item["attention_mask"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def prepare_data(self):
        """
        Загрузка и подготовка данных
        """
       
        self.dataset = load_dataset("wmt14", "ru-en")
        

        self.tokenized_train = self.dataset["train"].select(range(self.train_size)).map(self.preprocess)
        self.tokenized_eval = self.dataset["validation"].select(range(self.eval_size)).map(self.preprocess)
        self.tokenized_test = self.dataset["test"].select(range(self.test_size)).map(self.preprocess)

    def get_dataloaders(self):
        """
        Создание DataLoader'ов для обучения, валидации и тестирования
        """
        if self.tokenized_train is None:
            self.prepare_data()
            
       
        train_loader = DataLoader(
            self.tokenized_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        
        eval_loader = DataLoader(
            self.tokenized_eval, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        
        test_loader = DataLoader(
            self.tokenized_test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        
        return train_loader, eval_loader, test_loader

    def get_vocab_sizes(self):
        """
        Возвращает размеры словарей
        """
        return self.src_vocab_size, self.tgt_vocab_size

    def get_tokenizer(self):
        """
        Возвращает токенизатор
        """
        return self.tokenizer
    
