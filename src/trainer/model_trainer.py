import torch.optim as optim
from logger.logger import setup_logging
from logger.wandb import WandBWriter
from tqdm import tqdm

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
        self.writer.set_step((epoch-1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc = f"train {epoch - 1}", total = self.epoch_len)
        ):
            batch['inputs'] = batch['inputs'].to(self.device)
            batch['labels'] = batch['labels'].to(self.device)

            outputs = self.model(batch['inputs'])
            



    def _train_process(self):
        """
        Full train logic
        """

        pass

    
    def _evaluate_epoch(self, epoch):
        
        pass
    
    def _process_batch(self, batch):
        
        pass
    
    def _save_checkpoint(self, epoch, is_best=False):
       
        pass
    
    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        pass
