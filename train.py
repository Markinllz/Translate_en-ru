import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


from src.datasets.download_data import TranslationDataModule
from src.trainer.model_trainer import BaseTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging


warnings.filterwarnings("ignore", category=UserWarning)



@hydra.main(version_base=None, config_path= "src/configs", config_name= "config")
def main(config):

    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)


    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    else:
        device = "cpu"


    dataloaders = instantiate(config.dataset)

    train_loader, _ , test_loader = dataloaders.get_dataloaders()

    override_data = {
        "train": train_loader,
        "test" : test_loader
    }

    model = instantiate(config.model).to(device)
    logger.info(model)


    loss_function = instantiate(config.loss_function).to(device)


    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)


    trainer = BaseTrainer(
        model=model,
        criterion=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=override_data,
        logger=logger,
        writer=writer,
    )


    trainer.train()


if __name__ == "__main__":
    main()