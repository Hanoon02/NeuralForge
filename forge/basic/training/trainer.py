# Generic train/eval loops that any Module + loss + optimizer can plug into.
# TODO: train_one_epoch(model, dataloader, loss_fn, optimizer) - train mode; forward → loss → backward → optimizer.step(); return avg loss / metrics.
# TODO: evaluate(model, dataloader, loss_fn, metric_fns) - eval mode; no Parameter updates; return loss + metrics.
# TODO: fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, scheduler=None, checkpoint_dir=None) - Full multi-epoch loop wiring the above.
