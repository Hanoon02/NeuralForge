# LR schedules the trainer applies after each epoch/step (torch.optim.lr_scheduler analogues).
# TODO: class StepLR - Multiply lr by gamma every step_size epochs.
# TODO: class CosineAnnealingLR - Cosine decay from lr to eta_min over T_max.
# TODO: class LambdaLR - Arbitrary lr = base_lr * fn(epoch).
# TODO: scheduler.step() - Update optimizer lr in place.
# TODO: scheduler.state_dict() / load_state_dict() - Checkpoint schedule progress.
