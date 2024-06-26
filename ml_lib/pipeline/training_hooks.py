from datetime import datetime
from re import T
from textwrap import dedent
from typing import Optional, TYPE_CHECKING, Iterable, TypeVar, Protocol, Literal, overload

from dataclasses import dataclass, field
from io import StringIO
import itertools as it
from logging import info
import os
from pathlib import Path
import sqlite3
from time import sleep
from logging import getLogger
from torch.nn.modules.conv import F

from ml_lib.misc.basic import fill_default
from ml_lib.misc.data_structures import NotSpecified; logger = getLogger(__name__)

if TYPE_CHECKING:
    import matplotlib.axes
    from tqdm import tqdm
    from sqlalchemy.orm import Session
    from ml_lib.pipeline.experiment_tracking import Model as DBModel, Training_run as DBTraining_run, Training_step as DBTraining_step, NonBlockingStep, NonBlockingStepCache, CacheCheckpoint
    from trello import TrelloClient, List as TrelloList, Board as TrelloBoard, Card as TrelloCard, Checklist as TrelloChecklist


from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_

from ..environment import HasEnvironmentMixin, Scope, scopevar_of_str, str_of_scopevar
from ..misc import find_file
from .annealing_scheduler import AnnealingScheduler, get_scheduler
from ..register import Register


@dataclass
class TrainingHook(HasEnvironmentMixin):

    interval: int|None = 1
    absolutely_necessary: bool = True 
    """If set to False, exceptions raised in the hook will be ignored, 
    and printed as warnings instead. If True, they will crash the training"""
    # env: Environment = field(default_factory=Environment)
    __post_init__ = HasEnvironmentMixin.__init__

    def __call__(self):
        if self.interval is None: 
            self.env.environment.run_function(self._protected_hook) #this is for end hooks (only run once)
        n_iteration = self.env.iteration

        new_values = None
        if n_iteration % self.interval == 0:
            self.env.environment.run_function(self._protected_hook)

        if new_values is None:
            new_values = {}

    def _protected_hook(self):
        if self.absolutely_necessary: 
            self.hook()
            return
        try: 
            self.hook()
        except Exception as e: 
            logger.warning(f"Training hook {self} raised an exception {e}")

    def hook(self) -> Optional[dict]:
        raise NotImplementedError

    def setup(self):
        """Function called before running the training, Only global env variables are set up"""
        pass

    def set_state(self):
        """Set the state of the hook to the current state of the environment.
        Useful when the training is resumed from a checkpoint.
        Read the state of the environment and set the state of the hook accordingly.
        """
        pass

register = Register(TrainingHook)

class EndHook(TrainingHook):
    """End Hooks are by default not absolutely_necessary. """
    def __init__(self, absolutely_necessary=False):
        super().__init__(interval=None, absolutely_necessary=absolutely_necessary)
    def __call__(self):
        self.env.environment.run_function(self.hook) #no checking stuff because this is run once

class EndAndStepHook(EndHook):
    def __init__(self, interval=1, absolutely_necessary=True):
        TrainingHook.__init__(self, interval=interval)
    def __call__(self):
        if self.env.get("training_finished"):
            EndHook.__call__(self)
        else:
            TrainingHook.__call__(self)

@register
class LoggerHook(TrainingHook):
    variables: list[tuple[Scope, str]]
    absolutely_necessary = False

    def __init__(self, variables: list[str] = ["iteration", "loss"], interval=1):
        super().__init__(interval, absolutely_necessary=False)
        self.variables = [scopevar_of_str(v) for v in variables]

    def hook(self):
        s = StringIO()
        for scope, key in self.variables:
            value = self.env.get(key=key, scope=scope)
            if hasattr(value, "item") and value.numel() == 1:
                value = value.item()
            s.write(f"{str_of_scopevar(scope, key)}= {value}, ")

        info(s.getvalue())

@register
class CurveHook(TrainingHook):
    scope: Scope
    variable: str
    values: list

    def __init__(self, interval:int =1, variable="loss"):
        super().__init__(interval, absolutely_necessary=False)
        self.scope, self.variable = scopevar_of_str(variable)
        self.values = []
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        self.plt = plt
        self.mpl = mpl

    def hook(self):
        val = self.env.get(self.variable, self.scope)
        if hasattr(val, "item"):
            val = val.item()
        self.values.append(val)

    def draw(self, ax: "matplotlib.axes.Axes|None" = None): #todo: potentially output to file
        import numpy as np
        if ax is None:
            ax = self.plt.gca()
            assert isinstance(ax, self.mpl.axes.Axes)
        values = self.values
        ax.set_title(self.variable)
        ax.set_ylabel(self.variable)
        ax.plot(np.arange(len(values)) * self.interval, values)

@register
class KLAnnealingHook(TrainingHook):
    scope: Scope
    variable: str
    scheduler: AnnealingScheduler
    
    def __init__(self, variable="kl_coef",
                 scheduler: str|AnnealingScheduler = "constant",
                 beta_0 = None, T_0=None, T_1=None):
        super().__init__(interval=1)
        self.scope, self.variable = scopevar_of_str(variable)
        match scheduler:
            case AnnealingScheduler():
                assert beta_0 is None and T_0 is None and T_1 is None
                self.scheduler = scheduler
            case str():
                if beta_0 is None: beta_0 = 1.
                self.scheduler = get_scheduler(scheduler, beta_0, T_0, T_1)

    def hook(self):
        val = self.scheduler.step()
        self.env.record(self.variable, val, self.scope)

    def draw(self, ax: "matplotlib.axes.Axes|None" = None):
        self.scheduler.draw(ax=ax)

@register
class TqdmHook(TrainingHook):
    progressbar: Optional["tqdm"] = None
    last_known_epoch: int = 0
    absolutely_necessary = False

    def __init__(self, interval:int =1, tqdm=None):
        super().__init__(interval, absolutely_necessary=False)
        if tqdm is None:
            from tqdm.auto import tqdm
        self.tqdm = tqdm
        self.progressbar = None

    def hook(self):
        model = self.env.model
        if self.progressbar is None:
            self.reset_progressbar()
        assert self.progressbar is not None
        if self.env.epoch != self.last_known_epoch:
            self.last_known_epoch = self.env.epoch
            model_name = model.model_name or model.get_model_type()
            self.progressbar.set_description(f"Model {model_name} - Epoch {self.env.epoch}")
        self.progressbar.update()

    def reset_progressbar(self, initial: int = 0):
        totaliter =self.env.total_iter
        epoch = self.env.epoch
        model_name = self.env.model.model_name
        self.last_known_epoch = epoch
        self.progressbar = self.tqdm(total=totaliter, initial=initial, 
                                     desc=f"{model_name}: Epoch {epoch}", 
                                     smoothing=.1, dynamic_ncols=True)

    def set_state(self):
        step = self.env.iteration
        self.reset_progressbar(initial=step)

@register
class LogGradientInfo(TrainingHook):
    def hook(self):
        import torch
        model = self.env.model
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        all_gradients_1D = torch.cat([g.reshape(-1) for g in gradients])
        all_gradients_1D_abs = all_gradients_1D.abs()

        max_grad = all_gradients_1D_abs.max()
        min_grad = all_gradients_1D_abs.min()
        average_grad = all_gradients_1D_abs.mean() 

        self.env.max_grad = max_grad
        self.env.min_grad = min_grad
        self.env.average_grad = average_grad

        self.env.additional_log_vars = (self.env.get("additional_log_vars") or []) + [
                ((), "max_grad"), 
                ((), "min_grad"), 
                ((), "average_grad")
                ]



@register
class TensorboardHook(TrainingHook):
    
    def __init__(self, interval: int=1, *, tensorboard_dir:Optional[str] = None, 
                 run_name:str|None=None,  
                 log_vars = ["loss"]):
        super().__init__(interval, absolutely_necessary=False)
        if tensorboard_dir is None:
            if "TENSORBOARD" in os.environ: 
                tensorboard_path = Path(os.environ["TENSORBOARD"])
            else: 
                tensorboard_path = find_file([
                    Path("tensorboard"), Path("../tensorboard"), 
                    Path(f"{os.environ['HOME']}/tensorboard")])
            if tensorboard_path is None: tensorboard_path = Path("tensorboard")
        else : tensorboard_path = Path(tensorboard_dir)
        self.run_name = run_name
        self.tensorboard_path = tensorboard_path
        from torch.utils import tensorboard
        self.tensorboard = tensorboard
        self.log_vars = [scopevar_of_str(v) for v in log_vars]

    def setup(self):
        run_name = self.run_name
        if run_name is None: 
            run_name = self.env.model.model_name
        assert run_name is not None
        tensorboard_path = self.tensorboard_path / run_name
        self.writer = self.tensorboard.SummaryWriter(str(tensorboard_path))

    def hook(self):
        step = self.env.iteration
        loss_dict= dict()
        additional_log_vars = self.env.get("additional_log_vars") # additional variables registered as target for logging
        # todo if needed, add an option to disable those
        if additional_log_vars is None: additional_log_vars = []
        for scope, var in self.log_vars + additional_log_vars:
            loss_dict[var] = self.env.get(var, scope)
        self.writer.add_scalars('loss', loss_dict, step)

@register
class OptimizerHook(TrainingHook):
    """Probably the most important hook: runs the optimizer. 
    Without this hook the trainer won't train.

    No need to add it in the trainer configuration / call, it will be 
    automagically added.
    """
    def __init__(self, optimizer: Optimizer, clip_gradient: Optional[float] =None, interval: int=1):
        super().__init__(interval, absolutely_necessary=True)
        self.optimizer = optimizer
        self.clip_gradient = clip_gradient

    def hook(self):
        model = self.env.get("model")
        if self.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), self.clip_gradient)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

@register
class LRSchedulerHook(TrainingHook):
    def __init__(self, scheduler, interval: int=1):
        super().__init__(interval)
        self.scheduler = scheduler

    def hook(self):
        self.scheduler.step()

@register
class DatabaseHook(EndAndStepHook):
    database_session: "Session"
    model_id: int|None = None
    training_run_id: int|None = None

    training_run: "DBTraining_run|None" = None

    checkpoint_interval: int
    commit_interval: int

    loss_name: str = "loss"
    metrics: list[str] 

    flexible: bool = True
    commit_tries: int = 0 # the number of times we have failed to use session.commit() 
    #(because of database locking.). If >0, we should retry next iteration
    cache: "NonBlockingStepCache"

    def __init__(self, interval: int=1, *, database_session: "Session", checkpoint_interval: int = 100, commit_interval: int = 100, training_run_id: int|None=None, 
                 loss_name="loss", metrics=[], flexible: bool=True):
        super().__init__(interval, absolutely_necessary=False)
        from ml_lib.pipeline.experiment_tracking import NonBlockingStepCache
        assert checkpoint_interval%interval == 0
        assert commit_interval % interval == 0
        self.database_session = database_session

        self.checkpoint_interval = checkpoint_interval
        self.commit_interval = commit_interval
        self.training_run_id = training_run_id

        self.loss_name = loss_name
        self.metrics = metrics

        self.commit_tries = 0
        self.flexible = flexible
        self.cache = NonBlockingStepCache()

    def hook(self):
        from ml_lib.pipeline.experiment_tracking import Training_run as DBTraining_run, Training_step as DBTraining_step
        from sqlalchemy.exc import OperationalError
        step: int = self.env.iteration
        training_finished = self.env.get("training_finished") or False
        self.log_training_step_to_cache(training_finished)
        if training_finished or (step + 1) % self.commit_interval == 0 or self.commit_tries > 0:
            succeeded = self.cache.empty_into_db(self.database_session, allow_failure=self.flexible)
            if succeeded: self.commit_tries = 0
            else: self.handle_failure_to_commit(training_finished=training_finished)

    def setup(self):
        # from ..experiment_tracking import Training_run as DBTraining_run
        dbtraining_run = self.env.get("database_object")
        if dbtraining_run is None: raise ValueError("tried to create a database hook with no database object in the environment")

    def log_training_step_to_cache(self, is_last):
        from datetime import datetime
        from ml_lib.pipeline.experiment_tracking import NonBlockingStep, CacheCheckpoint
        #################### determine basic info
        step: int = self.env.iteration
        epoch: int = self.env.epoch
        if step is None:
            assert is_last
            step = self.env.total_iter - 1
            epoch = self.env.n_epochs - 1
        training_run: "DBTraining_run" = self.env.get("training_run_db")
        if training_run is None:
            raise ValueError("Tried to use DatabaseHook without a training_run_db object in the environment")
        training_run_id = training_run.id
            
        #################### determine metrics and loss
        loss = self.env.get(self.loss_name)
        step_time = datetime.now()
        if self.metrics:
            metrics = {metric: self.env.get(metric) for metric in self.metrics}
        elif (env_metrics:= self.env.get("metrics")) is not None:
            metrics = env_metrics
        else: metrics = {}

        #################### maybe checkpoint
        is_checkpointing_step = ((step + 1) % self.checkpoint_interval) == 0
        should_checkpoint = is_last or is_checkpointing_step

        if should_checkpoint:
            model = self.env.model
            checkpoint = CacheCheckpoint.from_model(model, is_last=is_last,
                    session=self.database_session)
        else: checkpoint = None
        
        #################### record into the cache
        training_step: NonBlockingStep= NonBlockingStep(
            training_run_id=training_run_id,
            step=step, 
            step_time=step_time, 
            loss=loss,
            epoch=epoch, 
            metrics=metrics, 
            checkpoint=checkpoint
            )
        self.cache.add(training_step)


    def handle_failure_to_commit(self, training_finished):
        if training_finished:
            for try_n in it.count():
                logger.warning(f"Failed to checkpoint at the end of training. retrying for the {try_n}th time")
                try: 
                    self.cache.empty_into_db(self.database_session, allow_failure=False) 
                    #allow_failure = False re-raises, so we're able to catch and print the error
                except Exception as e: 
                    logger.warning(f"Got exception {e}")
                    sleep(1)
            return
        if (self.commit_tries - 1) % 10 == 0 :
            logger.warning(f"failed commiting to db for {self.commit_tries} consecutive iterations")
        self.commit_tries += 1 # makes us retry next iteration



@register
class SlackHook(EndHook):
    """Sends you a slack message when a model finished training
    Sadly, does'nt work for now, i can't manage to figure out the slack API"""
    token: str
    channel: str

    def __init__(self, *, token:str|None=None, channel:str|None=None, ):
        super().__init__()
        if token is None:
            token = os.environ["SLACK_TOKEN"]
        if channel is None:
            channel = os.environ["SLACK_CHANNEL"]
        if channel.startswith("#"):
            import requests
            response = requests.get("https://slack.com/api/conversations.list", params={
                "token": token
            })
            assert response.ok
            response = response.json()
            assert response["ok"]
            channels = response["channels"]
            channel_name = channel[1:]
            for channel_i in channels:
                if channel_i["name"] == channel_name:
                    channel = channel_i["id"]
                    break
            else:
                logger.error(f"SlackHook: Channel {channel_name} not found, will not send slack messages")


        assert channel is not None
        self.token = token
        self.channel = channel

    def hook(self):
        import requests
        model = self.env.get("model")
        model_name = model.model_name
        if model_name is not None:
            train_info = model_name
        else:
            import sys
            train_info = " ".join(sys.argv)

        requests.post("https://slack.com/api/chat.postMessage", data={
            "token": self.token,
            "channel": self.channel,
            "text": f"Training finished for {train_info}"
        })


class _HasName(Protocol):
    name: str
T_HasName = TypeVar("T_HasName", bound=_HasName)

@register
class TrelloHook(EndHook):
    """Updates trello cards. Also doubles as a notification system by adding trello events."""

    api_key: str
    token: str

    client: "TrelloClient"

    board: "TrelloBoard"
    card: "TrelloCard|None" = None
    ongoing_list_name: str
    finished_list_name: str

    ongoing_list: "TrelloList"
    finished_list: "TrelloList"


    def __init__(self, *, 
                 api_key: str|NotSpecified=NotSpecified(), 
                 token: str|NotSpecified=NotSpecified(), 
                 board:str|NotSpecified= NotSpecified(), 
                 ongoing_list: str="Running experiments", finished_list: str="Finished Running"):
        from trello import TrelloClient
        self.api_key = fill_default(api_key, env_variable="TRELLO_API_KEY", name="trello api key")
        self.token = fill_default(token, env_variable="TRELLO_TOKEN", name="trello token")
        board_name = fill_default(board, env_variable="TRELLO_BOARD", name="trello board")

        self.client = TrelloClient(api_key=self.api_key, token=self.token)
        self.board = self.find_name(self.client.list_boards(), board_name)

        self.ongoing_list_name = ongoing_list
        self.ongoing_list = self.get_list(ongoing_list)
        self.finished_list_name = finished_list
        self.finished_list = self.get_list(finished_list)
        self.card = None

    def setup(self):
        model_card = self.find_model_card()
        if model_card is None: model_card = self.make_model_card()
        self.card = model_card
        self.card.comment(f"Started training on machine {os.environ.get('HOSTNAME', 'unknown')} at {datetime.now().isoformat()}")

    def get_list(self, name) -> "TrelloList":
        return self.find_name(self.board.list_lists(), name)

    def find_model_card(self,):
        model_name = self.env.model.model_name
        for l in [self.finished_list, self.ongoing_list]:
            model_card = self.find_name(l.list_cards_iter(), model_name, allowNone=True)
            if model_card: return model_card
        return None

    def make_model_card(self, ongoing=True):
        model = self.env.model
        model_name = model.model_name
        training_parameters = self.env.training_parameters

        model_description= (
                "## Model parameters\n\n"
                f"```python\n{str(model)}```\n\n"
                "## Training parameters\n\n"
                f"```python{str(training_parameters)}```"
        )
        if ongoing: l = self.ongoing_list
        else: l = self.finished_list

        card = l.add_card(
            name = model_name, 
            desc = model_description, 
        )

        _ = card.add_checklist("status", ["Train", "Evaluate"])
        return card

    def get_status_checklist(self,  card: "TrelloCard", add=True) -> "TrelloChecklist|None":
        status_checklist = self.find_name(card.checklists, "status", allowNone=True)
        if status_checklist is not None:
            return status_checklist
        if add:
            status_checklist = card.add_checklist("status", ["Train", "Evaluate"])
        return status_checklist

    def hook(self):
        card = self.card
        assert card is not None
        checklist = self.get_status_checklist(card, add=True)
        assert checklist is not None
        checklist.set_checklist_item("Train", checked=True)
        card.change_list(self.finished_list.id)

        
    @overload
    @staticmethod
    def find_name(l: Iterable[T_HasName], name: str, allowNone:Literal[False]=False) -> T_HasName:
        ...

    @overload
    @staticmethod
    def find_name(l: Iterable[T_HasName], name: str, allowNone:Literal[True]) -> T_HasName|None:
        ...

    @staticmethod
    def find_name(l: Iterable[T_HasName], name: str, allowNone=False) -> T_HasName|None:
        """takes a list of elements with """
        for elt in l:
            if elt.name == name:#type:ignore
                return elt
        if not allowNone:
            raise IndexError(f"didn't find {name} in {l}")
        return None

