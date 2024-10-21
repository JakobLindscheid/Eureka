import wandb
from rl_games.common.algo_observer import AlgoObserver

from isaacgymenvs.utils.utils import retry
from isaacgymenvs.utils.reformat import omegaconf_to_dict


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.successes = []
        self.gpt_rewards = []
        self.gt_rewards = []

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        cfg = self.cfg
        # ema = cfg.task.env.actionsMovingAverage

        # wandb_unique_id = f'unique_id_{experiment_name}' + f'_ema={ema}'
        wandb_unique_id = f'{experiment_name}' # + f'_ema={ema}'
        print(f'Wandb using unique id {wandb_unique_id}')

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception, ))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method='fork'),
            )
            try:
                wandb.save("env.py")
            except:
                pass

            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print('wandb running directory........', wandb.run.dir)

        print('Initializing WandB...')
        try:
            init_wandb()
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), 'AlgoObserver expects dict info'
        if "consecutive_successes" in infos:
            self.successes.append(infos["consecutive_successes"])

        if "gpt_reward" in infos:
            self.gpt_rewards.append(infos["gpt_reward"])

        if "gt_reward" in infos:
            self.gt_rewards.append(infos["gt_reward"])

    def after_print_stats(self, frame, epoch_num, total_time):
        wandb.log({
            k: sum(v) / len(v) 
            for k, v in [('success', self.successes), ('gpt_reward', self.gpt_rewards), ('gt_reward', self.gt_rewards)] 
            if len(v) > 0
        })