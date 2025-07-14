import torch.optim as optim
from logger.logger import setup_logging
from logger.wandb import WandBWriter
from tqdm import tqdm
from abc import abstractmethod
import torch

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
        self.test_loader = dataloaders["test"]
        self.epoch_len = len(self.train_loader)
    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model key board interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e
    
    def _train_epoch(self, epoch):

        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """

        self.is_train = True
        self.model.train()
        total_loss = 0
        batch_count = 0
        self.writer.set_step((epoch-1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc = f"train {epoch - 1}", total = self.epoch_len)
        ):


            #Batch to device
            batch['input_ids'] = batch['input_ids'].to(self.device)
            batch['labels'] = batch['labels'].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)


            #Model prediction
            outputs = self.model(
                batch['input_ids'],
                attention_mask=attention_mask
                )

            #Count loss
            loss = self.criterion(outputs.logits, batch['labels'])

            #Gradient step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            #Compute losses
            total_loss += loss.item()
            batch_count += 1


            if (batch_idx % self.log_step) == 0:
                avg_loss = total_loss / batch_count

                #WandB step
                current_step = (epoch - 1) * self.epoch_len + batch_idx
                self.writer.set_step(current_step, "train")


                #Logging to WandB
                self.writer.add_scalar("loss", avg_loss)
                self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'])


                #Logging to console
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss = {avg_loss}")


        epoch_loss = total_loss / batch_count if batch_count > 0 else 0.0
            # Evaluation
        val_loss = self._evaluate_epoch(epoch)
            #Return values
        return {"loss": epoch_loss, "val_loss": val_loss}
    
    def _train_process(self):
        """
        Full train logic
        """
        for epoch in range(self.num_epochs):
            epoch_loss, val_loss = self._train_epoch(epoch)


            if epoch % 5 == 0:
                self._save_checkpoint(save_best=False,only_best=False)


            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint(epoch, save_best=True, only_best=True)
                self.logger.info(f"New best model! Val Loss: {val_loss:.4f}")

    
    def _evaluate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        batch_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(self.test_loader, desc = f"Evaluation {epoch - 1}")
            ):
                #Move to device
                batch['input_ids'] = batch['input_ids'].to(self.device)
                batch['labels'] = batch['labels'].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)



                #Model predtiction


                outputs = self.model(
                    batch['input_ids'],
                    attention_mask = attention_mask
                    )


                #Loss
                loss = self.criterion(outputs, batch['labels'])


                #Updating
                batch_count += 1
                val_loss += loss.item()


        avg_val_loss = val_loss / batch_count if batch_count > 0 else 0.0
        self.writer.set_step(epoch * self.epoch_len, "val")
        self.writer.add_scalar("loss", avg_val_loss)
        
        return avg_val_loss



    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        #Create state
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        
    
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        
        #Path to checkpoints
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        #Save all
        if not (only_best and save_best):
            torch.save(state, filename)
            if hasattr(self, 'writer') and self.writer:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            if hasattr(self, 'logger'):
                self.logger.info(f"Saving checkpoint: {filename} ...")
        #Save onlut the best
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if hasattr(self, 'writer') and self.writer:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            if hasattr(self, 'logger'):
                self.logger.info("Saving current best: model_best.pth ...")
            
        
