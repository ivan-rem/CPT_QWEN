from transformers.trainer_callback import TrainerCallback
import wandb

class LogTrainTokensCallback(TrainerCallback):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def on_step_end(self, args, state, control, **kwargs):
        wandb.log({
            "total_tokens_seen": state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size * self.max_seq_len
        })
        return control