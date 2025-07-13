from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import MarianTokenizer

class TranslationDataModule:
    def __init__(
        self,
        batch_size=4,
        max_length=64,
        train_size=10000,
        eval_size=1000,
        test_size=1000,
        model_name="Helsinki-NLP/opus-mt-en-ru"
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_size = train_size
        self.eval_size = eval_size
        self.test_size = test_size
        self.model_name = model_name

        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.src_vocab_size = self.tokenizer.vocab_size
        self.tgt_vocab_size = self.tokenizer.vocab_size

        self.dataset = None
        self.tokenized_train = None
        self.tokenized_eval = None
        self.tokenized_test = None

    def preprocess(self, example):
        src = example["translation"]["ru"]
        tgt = example["translation"]["en"]
        src_enc = self.tokenizer(src, truncation=True, padding="max_length", max_length=self.max_length)
        tgt_enc = self.tokenizer(tgt, truncation=True, padding="max_length", max_length=self.max_length)
        return {
            "input_ids": src_enc["input_ids"],
            "attention_mask": src_enc["attention_mask"],
            "labels": tgt_enc["input_ids"]
        }

    def prepare_data(self):
        self.dataset = load_dataset("wmt14", "ru-en")
        self.tokenized_train = self.dataset["train"].select(range(self.train_size)).map(self.preprocess)
        self.tokenized_eval = self.dataset["validation"].select(range(self.eval_size)).map(self.preprocess)
        self.tokenized_test = self.dataset["test"].select(range(self.test_size)).map(self.preprocess)

    def get_dataloaders(self):
        if self.tokenized_train is None:
            self.prepare_data()
        train_loader = DataLoader(self.tokenized_train, batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(self.tokenized_eval, batch_size=self.batch_size)
        test_loader = DataLoader(self.tokenized_test, batch_size=self.batch_size)
        return train_loader, eval_loader, test_loader
    