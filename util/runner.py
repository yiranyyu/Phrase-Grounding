import numpy as np
import torch as th
from torch.utils.data import DataLoader

from dataset.flickr30k_entities import Flickr30kEntities
from util.visdom import vis_init, vis_create
from ignite.utils import convert_tensor
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan, Timer
from ignite.exceptions import NotComputableError
from models import bert
from ignite.metrics import Metric
from util import logging
from util.utils import set_random_seed


class BCELoss(Metric):
    """Compute mean entity BCE loss per batch and epoch.
    """

    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(BCELoss, self).__init__(output_transform=output_transform)
        self._loss_fn = loss_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        (y_pred, y) = output[:2]
        if len(output) == 3:
            average_loss, E = output[2]
        else:
            average_loss, E = self._loss_fn(y_pred, y)

        assert average_loss.dim() == 0, "loss_fn() did not return the average loss."

        self._sum += average_loss.item() * E
        self._num_examples += E

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "Loss must have at least one example before it can be computed."
            )
        return self._sum / self._num_examples


class EntityRecall(Metric):
    """Compute top K entity recalls per batch and epoch.
    """

    def __init__(self, topk=[1, 5, 10], typeN=len(Flickr30kEntities.ETypes), output_transform=lambda x: x):
        self.topk = topk if isinstance(topk, list) else [topk]
        self.typeN = typeN
        super(EntityRecall, self).__init__(output_transform)

    def reset(self):
        self._N = 0.0
        self._TPs = th.tensor([0.0] * (len(self.topk) + 1))
        # self._bound = 0.0
        self._typeTPs = th.zeros(self.typeN)
        self._typeN = th.zeros(self.typeN)

    def update(self, output):
        (y_pred, y) = output[:2]
        N, TPs, (typeTPs, typeN) = bert.recall(y_pred, y, topk=self.topk, typeN=self.typeN)
        self._N += N
        self._TPs += TPs
        self._typeTPs += typeTPs
        self._typeN += typeN

    def compute(self):
        if self._N == 0:
            raise NotComputableError(
                "Loss must have at least one example before it can be computed."
            )
        TPs = self._TPs / self._N
        typeTPs = self._typeTPs / (self._typeN + (self._typeN == 0).float())
        logging.debug(self._typeN.tolist())
        return TPs.tolist(), typeTPs.tolist()


def prepare_batch(batch, device=None, non_blocking=True):
    """Prepare batch for training: pass to a device with options
    """
    # batch: (features, spatials, mask), tokens, indices, target
    batch = [
        convert_tensor(field, device=device, non_blocking=non_blocking)
        for field in batch
    ]
    return batch


def create_supervised_trainer(
        model,
        loss_fn,
        optim,
        grad_acc_steps=1,
        device=None,
        non_blocking=False,
        prepare_batch=prepare_batch,
        output_transform=lambda x, y, y_pred, loss: loss.item(),
):
    """
    Factory function for creating a trainer for supervised models.
    No metrics are attached for training.
    Custom metrics must be attached separately.

    Args:
        model (`torch.nn.Module`): the model to train that is already parallelized if necessary
        optim (`torch.optim.Optimizer`): the optimizer to use
        optim_steps (int): intermediate number of steps to accumulate gradients
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (Callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
    Returns:
        Engine: a trainer engine with supervised update function
    """

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss, E = loss_fn(y_pred, y)
        loss = loss / grad_acc_steps
        loss.backward()
        if engine.state.iteration % grad_acc_steps == 0:
            optim.step()
            optim.zero_grad()

        return output_transform(x, y, y_pred, (loss * grad_acc_steps, E))

    return Engine(_update)


def gen_dataloader(cfg, split, bs, shuffle=False):
    """Create a data loader for the specified dataset.
    """
    if cfg.dataset == "flickr30k_entities":
        ds = Flickr30kEntities(
            split,
            path=cfg.data / "flickr30k_entities",
            tokenization=cfg.tok,
            max_tokens=cfg.max_tokens,
            max_entities=cfg.max_entities,
            max_rois=cfg.max_rois,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    num_workers = cfg.num_workers or max(th.get_num_threads() // 2, 2)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers)


def setup_model(cfg):
    if cfg.arch == "bert":
        from models.bert import IBertConfig, BertForGrounding
        cfgI = IBertConfig(
            hidden_size=cfg.bert_img_hidden_size,
            num_hidden_layers=cfg.bert_img_layers,
            num_attention_heads=cfg.bert_img_heads,
            intermediate_size=cfg.bert_img_intermediate_size,
            hidden_dropout_prob=cfg.bert_img_hidden_dp,
            attention_probs_dropout_prob=cfg.bert_img_attention_dp,
            spatial=cfg.bert_img_spatial,
        )
        model = BertForGrounding(cfgI)
    else:
        raise ValueError(f"Unknown arch: {cfg.arch}")

    if cfg.resume:
        model.load(cfg.resume)
        logging.info(f"{cfg.resume}")

    from ml import nn

    return nn.parallelize(model, device_ids=cfg.gpu)


def prepare_train(cfg):
    """Prepare for data, model, task
    """

    # Dataset and DataLoader
    bs0, bs1 = cfg.bs[0] // cfg.grad_acc_steps, cfg.bs[1]
    train_loader = gen_dataloader(cfg, cfg.split[0], bs=bs0, shuffle=True)
    dev_loader = gen_dataloader(cfg, cfg.split[1], bs=bs1, shuffle=False)
    logging.info(
        f"Loaded train/dev datasets of size {len(train_loader)}/{len(dev_loader)}"
    )

    # Model, optimizer and parameters
    model, device = setup_model(cfg)
    if cfg.optim == "adam":
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        params = list(model.named_parameters())
        gparams = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.adam_wd,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        from ml.nlp import bert

        num_opt_steps = len(train_loader.dataset) // cfg.bs[0] * cfg.epochs
        optim = bert.BertAdam(
            gparams,
            lr=cfg.lr,
            warmup=cfg.warmup,
            t_total=num_opt_steps,
            b1=cfg.adam_beta1,
            b2=cfg.adam_beta2,
            e=cfg.adam_eps,
            weight_decay=cfg.adam_wd,
            max_grad_norm=cfg.max_grad_norm,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optim}")

    logging.info(f"Set up {cfg.model} and optimizer {cfg.optim}")
    return (train_loader, dev_loader), model, optim, device


def prepare_test(cfg):
    test_loader = gen_dataloader(cfg, cfg.split, bs=cfg.bs, shuffle=False)
    logging.info(f"Loaded {cfg.split} dataset of size {len(test_loader)}")

    model, device = setup_model(cfg)
    logging.info(f"Set up {cfg.model}")
    return test_loader, model, device


def train(cfg, logging, vis=None):
    dataloaders, model, optim, device = prepare_train(cfg)
    trainer = create_supervised_trainer(
        model,
        bert.BCE_with_logits,
        optim,
        grad_acc_steps=cfg.grad_acc_steps,
        device=device,
        output_transform=lambda x, y, y_pred, loss: (y_pred, y, loss),
    )
    evaluator = create_supervised_evaluator(
        model, device=device, output_transform=lambda x, y, y_pred: (y_pred, y)
    )

    topk = [1, 5, 10]
    metrics = {
        "bce": BCELoss(loss_fn=bert.BCE_with_logits),
        "recall": EntityRecall(topk=topk),
    }
    for name, metric in metrics.items():
        metric.attach(trainer, name)
        metric.attach(evaluator, name)

    batchTimer = Timer(average=True)
    batchTimer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )
    epochTimer = Timer(average=True)
    epochTimer.attach(evaluator)

    if vis is not None:
        train_loss_win = vis_create(vis, "Training Loss", "#Iterations", "Loss")
        train_recall_win = vis_create(
            vis,
            "Training Recall",
            "#Iterations",
            "Recall",
            legend=[f"R@{k}" for k in topk] + ["bound"],
            npts=len(topk) + 1,
        )
        train_type_recall_win = vis_create(
            vis,
            "Training Type Recall",
            "#Iterations",
            "Recall",
            legend=Flickr30kEntities.ETypes,
            npts=len(Flickr30kEntities.ETypes),
        )
        val_loss_win = vis_create(vis, "Validation Loss", "#Epochs", "Loss")
        val_recall_win = vis_create(
            vis,
            "Validation Recall",
            "#Epochs",
            "Recall",
            legend=[f"R@{k}" for k in topk] + ["bound"],
            npts=len(topk) + 1,
        )
        val_type_recall_win = vis_create(
            vis,
            "Validation Type Recall",
            "#Iterations",
            "Recall",
            legend=Flickr30kEntities.ETypes,
            npts=len(Flickr30kEntities.ETypes),
        )

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    @trainer.on(Events.ITERATION_COMPLETED)
    def trainer_iteration_completed(engine):
        epoch, iteration, batches = (
            engine.state.epoch,
            engine.state.iteration,
            len(engine.state.dataloader),
        )
        iteration_e = iteration % batches
        iteration_e = batches if iteration_e == 0 else iteration_e
        if iteration_e % cfg.log_interval == 0 or iteration_e == batches:
            loss = metrics["bce"].compute()
            recall, typeRecall = metrics["recall"].compute()
            recalls = tuple(round(100 * v, 2) for v in recall)
            typeRecalls = tuple(round(100 * v, 2) for v in typeRecall)
            throughput = 1 / batchTimer.value()
            logging.info(
                f"[{epoch}/{cfg.epochs}][{iteration_e}/{batches}] "
                f"training loss={loss:.4f}, recalls={recalls}, per type={typeRecalls}, throughput={throughput:.2f} it/s"
            )
            if vis is not None:
                vis.line(
                    X=[iteration],
                    Y=np.array([loss]),
                    win=train_loss_win,
                    update="append",
                )
                vis.line(
                    X=np.array([iteration]),
                    Y=np.array([recalls]),
                    win=train_recall_win,
                    update="append",
                )
                vis.line(
                    X=np.array([iteration]),
                    Y=np.array([typeRecalls]),
                    win=train_type_recall_win,
                    update="append",
                )

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_epoch_completed(engine):
        epoch = engine.state.epoch
        evaluator.run(dataloaders[1], 1)
        metrics = evaluator.state.metrics
        loss = metrics["bce"]
        recall, typeRecall = metrics["recall"]
        recalls = tuple(round(100 * v, 2) for v in recall)
        typeRecalls = tuple(round(100 * v, 2) for v in typeRecall)
        logging.info(
            f"[{epoch}/{cfg.epochs}] "
            f"validation loss={loss:.4f}, recalls={recalls}, per type={typeRecalls}, time={epochTimer.value():.3f}s"
        )
        if vis is not None:
            vis.line(
                X=np.array([epoch]),
                Y=np.array([loss]),
                win=val_loss_win,
                update="append",
            )
            vis.line(
                X=np.array([epoch]),
                Y=np.array([recalls]),
                win=val_recall_win,
                update="append",
            )
            vis.line(
                X=np.array([epoch]),
                Y=np.array([typeRecalls]),
                win=val_type_recall_win,
                update="append",
            )

    checkpointer = ModelCheckpoint(
        cfg.save,
        "grounding",
        score_function=lambda engine: engine.state.metrics["recall"][0][0],
        score_name="recall",
        n_saved=cfg.nsaved,
    )
    stopper = EarlyStopping(
        trainer=trainer,
        patience=10,
        score_function=lambda engine: engine.state.metrics["recall"][0][0],
    )
    evaluator.add_event_handler(
        Events.COMPLETED,
        checkpointer,
        {"model": model.module if hasattr(model, "module") else model},
    )
    evaluator.add_event_handler(Events.COMPLETED, stopper)
    trainer.run(dataloaders[0], cfg.epochs)


def test(cfg):
    dataloader, model, device = prepare_test(cfg)

    topk = [1, 5, 10]
    metrics = {
        "bce": BCELoss(loss_fn=bert.BCE_with_logits),
        "recall": EntityRecall(topk=topk),
    }
    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        device=device,
        output_transform=lambda x, y, y_pred: (y_pred, y),
    )

    epochTimer = Timer(average=True)
    epochTimer.attach(evaluator)

    @evaluator.on(Events.COMPLETED)
    def test_completed(engine):
        loss = engine.state.metrics["bce"]
        recall, typeRecall = engine.state.metrics["recall"]
        recalls = tuple(round(100 * v, 2) for v in recall)
        typeRecalls = tuple(round(100 * v, 2) for v in typeRecall)
        logging.info(
            f"Test loss={loss:.4f}, recalls={recalls}, per type={typeRecalls}, time={epochTimer.value():.3f}s"
        )

    evaluator.run(dataloader, 1)


def run(cfg):
    logging.info(f"{cfg.cmd.capitalize()} model: {cfg.model}")
    logging.info(cfg)
    set_random_seed(cfg.seed, deterministic=True)

    if cfg.cmd == "train":
        # vis = vis_init(env=f"{cfg.cmd}-{cfg.model}")
        vis = None
        train(cfg, logging, vis)
    elif cfg.cmd == "test":
        logging.info('test')
        test(cfg)
