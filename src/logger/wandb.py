from datetime import datetime
import wandb

import numpy as np
import pandas as pd


class WandBWriter:

    def __init__(
            self,
            logger,
            project_config,
            project_name,
            entity = None,
            run_id = None,
            run_name = None,
            mode = "online",
            **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.

        Args:
            logger (Logger): logger that logs output.
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            entity (str | None): name of the entity inside experiment
                tracker. Used if you work in a team.
            run_id (str | None): the id of the current run.
            run_name (str | None): the name of the run. If None, random name
                is given.
            mode (str): if online, log data to the remote server. If
                offline, log locally.
        """
        try:
            import wandb
            wandb.login()
            self.run_id = run_id

            wandb.init(
                project = project_name,
                entity = entity,
                config = project_config,
                name = run_name,
                resume = "allow",
                id = self.run_id,
                mode = mode,
                save_code = kwargs.get("save_code", False)

            )
            self.wandb = wandb
        except ImportError:
            logger.warning("Install WandB")


        
        self.step = 0

        self.mode = ""

        self.timer = datetime.now()



    def set_step(self, step, mode = "train"):
        """
        Define cuurent step and mode for the tracker

        Calculates the difference betweeen method calls to monitor
        training/evaluation speed.


        Args:
            step (int): current step.
            mode (str): current mode("train", "val")
        """


        self.mode = mode

        previous_step = self.step
        
        self.step = step

        if step == 0:
            self.timer == datetime.now()

        else:
            duration = datetime.now() - self.timer

            self.add_scalar("steps_per_sec", (self.step - previous_step) / duration.total_seconds())


            self.timer = datetime.now()

    def _object_name(self, object_name):


        return f"{object_name}_{self.mode}"
    

    def add_checkpoint(self, checkpoint_path, save_dir):

        """
        Log checkpoints to the experiment tracker.

        The checkpoints will be available in the files section
        inside the run_name dir.

        Args:
            checkpoint_path (str): path to the checkpoint file.
            save_dir (str): path to the dir, where checkpoint is saved.
        """

        self.wandb.save(checkpoint_path, base_path = save_dir)

    
    def add_scalar(self, scalar_name, scalar):
         """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
         

         self.wandb.log(
             {
                 self._object_name(scalar_name): scalar
             },
             step = self.step,
         )


    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.

        Args:
            scalars (dict): dict, containing scalar name and value.
        """

        self.wandb.log(
            {
                self._object_name(scalar_name): scalar
                for scalar_name, scalar in scalars.items()
            },
            step = self.step,
        )


    def add_text(self, text_name, text):
         """
        Log text to the experiment tracker.

        Args:
            text_name (str): name of the text to use in the tracker.
            text (str): text content.
        """
         
         self.wandb.log(
            {self._object_name(text_name): self.wandb.Html(text)},
            step = self.step,
         )




    


