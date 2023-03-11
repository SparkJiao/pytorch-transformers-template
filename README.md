# pytorch-transformers-template
This is a template for fast prototype relying on transformers, hydra, fairscale and deepspeed, etc.

### Update Notes
#### 2023/02/11
- Unify torch vanilla FSDP training with fairscale FSDP into single trainer.
- Add some post processors as example for reference.
- Add a Trie based logits processor as example.
- Optimize the evaluator.
- Add trainer supporting multiple training dataset.
- Add support to different learning schedulers.

### TODO List
- ~~Multiple training set.~~
- Use wandb to replace naive tensorboard.
- Resume training from checkpoint.
  - Different setup for different engines, e.g., fairscale and deepspeed, or vanilla pytorch.
