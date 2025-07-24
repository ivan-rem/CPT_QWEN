from transformers.trainer_callback import TrainerCallback
import wandb

class LogTrainTokensCallback(TrainerCallback):
    def __init__(self):
        self.total_tokens = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            # Rough estimate: batch_size * seq_len * world_size (not exact with GA)
            # You can compute more accurately if you track it in the dataset.
            if wandb.run is not None:
                wandb.log({"total_tokens_seen": state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size * args.max_seq_length})
        return control