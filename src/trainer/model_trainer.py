import torch.optim as optim
from src.logger.logger import setup_logging
from src.logger.wandb import WandBWriter
from tqdm import tqdm
from abc import abstractmethod
import torch
from pathlib import Path

class BaseTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        num_epochs = 25,
    ):
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.num_epochs = num_epochs
        self.writer = writer
        self.train_loader = dataloaders["train"]
        self.eval_loader = dataloaders.get("eval", dataloaders["test"])
        self.test_loader = dataloaders["test"]
        self.epoch_len = len(self.train_loader)
        
        self._last_epoch = 0
        self.best_loss = float('inf')
        self.checkpoint_dir = Path(config.trainer.save_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model key board interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e
    
    def _train_epoch(self, epoch):
        self.is_train = True
        self.model.train()
        total_loss = 0
        batch_count = 0
        self.writer.set_step(epoch * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc = f"train {epoch}", total = self.epoch_len)
        ):
            batch['input_ids'] = batch['input_ids'].to(self.device)
            batch['labels'] = batch['labels'].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            src_input = batch['input_ids'][:, :-1]
            tgt_input = batch['labels'][:, :-1]
            tgt_output = batch['labels'][:, 1:]
            src_mask = attention_mask[:, :-1].transpose(0, 1)

            outputs = self.model(
                src_input,
                tgt_input,
                src_mask=src_mask
                )

            loss = self.criterion(outputs.logits, tgt_output)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            batch_count += 1

            if (batch_idx % self.log_step) == 0:
                avg_loss = total_loss / batch_count

                current_step = epoch * self.epoch_len + batch_idx
                self.writer.set_step(current_step, "train")

                self.writer.add_scalar("loss", avg_loss)
                self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'])

                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss = {avg_loss}")

        epoch_loss = total_loss / batch_count if batch_count > 0 else 0.0
        val_loss = self._evaluate_epoch(epoch)
        return epoch_loss, val_loss
    
    def _train_process(self):
        for epoch in range(self.num_epochs):
            epoch_loss, val_loss = self._train_epoch(epoch)
            self._last_epoch = epoch

            if epoch % 5 == 0:
                self._save_checkpoint(epoch, save_best=False, only_best=False)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint(epoch, save_best=True, only_best=True)
                self.logger.info(f"New best model! Val Loss: {val_loss:.4f}")

    
    def _evaluate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        batch_count = 0
        examples_logged = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(self.eval_loader, desc = f"Evaluation {epoch}")
            ):
                batch['input_ids'] = batch['input_ids'].to(self.device)
                batch['labels'] = batch['labels'].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                src_input = batch['input_ids'][:, :-1]
                tgt_input = batch['labels'][:, :-1]
                tgt_output = batch['labels'][:, 1:]
                src_mask = attention_mask[:, :-1].transpose(0, 1)

                outputs = self.model(
                    src_input,
                    tgt_input,
                    src_mask = src_mask
                    )

                loss = self.criterion(outputs.logits, tgt_output)

                batch_count += 1
                val_loss += loss.item()
                
                if examples_logged < 3 and batch_idx % 10 == 0:
                    self._log_translation_examples(batch, outputs, batch_idx)
                    examples_logged += 1

        avg_val_loss = val_loss / batch_count if batch_count > 0 else 0.0
        self.writer.set_step(epoch * self.epoch_len, "val")
        self.writer.add_scalar("loss", avg_val_loss)
        
        return avg_val_loss
    
    def _log_translation_examples(self, batch, outputs, batch_idx):
        try:
            tokenizer = self.eval_loader.dataset.dataset.tokenizer
            
            for i in range(min(2, batch['input_ids'].size(0))):
                src_tokens = batch['input_ids'][i]
                tgt_tokens = batch['labels'][i]
                pred_tokens = outputs.logits[i].argmax(dim=-1)
                
                src_text = tokenizer.decode(src_tokens, skip_special_tokens=True)
                tgt_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                
                example_text = f"**Input (EN):** {src_text}\n**Target (RU):** {tgt_text}\n**Prediction (RU):** {pred_text}"
                
                self.writer.add_text(f"translation_example_{batch_idx}_{i}", example_text)
                
        except Exception as e:
            self.logger.warning(f"Could not log translation examples: {e}")

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if hasattr(self, 'writer') and self.writer:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            if hasattr(self, 'logger'):
                self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if hasattr(self, 'writer') and self.writer:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            if hasattr(self, 'logger'):
                self.logger.info("Saving current best: model_best.pth ...")