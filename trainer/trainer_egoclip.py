from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from base import Multi_BaseTrainer_dist
from model.loss import EgoNCE, MaxMarginRankingLoss
from model.model import sim_matrix
from utils import (
    inf_loop,
    move_video_data_to_device,
    save_results,
    tokenize_and_move_to_gpu,
)


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
        )


class Multi_Trainer_dist(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self,
        args,
        model,
        loss,
        metrics,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
        writer=None,
        visualizer=None,
        tokenizer=None,
        max_samples_per_epoch=50000,
    ):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        # self.writer = writer

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            # if self.writer is not None:
            #     self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if "video_neg" in data.keys():  # w/ negative sampling
                    data["text"] = data["text"] + data["text_neg"]
                    data["video"] = torch.cat(
                        (data["video"], data["video_neg"]), axis=0
                    )
                    data["noun_vec"] = torch.cat(
                        (data["noun_vec"], data["noun_vec_neg"]), axis=0
                    )
                    data["verb_vec"] = torch.cat(
                        (data["verb_vec"], data["verb_vec_neg"]), axis=0
                    )

                if self.tokenizer is not None:
                    data["text"] = self.tokenizer(
                        data["text"], return_tensors="pt", padding=True, truncation=True
                    )
                data["text"] = {
                    key: val.to(self.device) for key, val in data["text"].items()
                }
                data["video"] = data["video"].to(self.device)
                n_embeds = data["noun_vec"].to(self.device)
                v_embeds = data["verb_vec"].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    text_embeds, video_embeds = self.model(data)
                    video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
                    text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)
                    n_embeds = self.allgather(n_embeds, self.n_gpu, self.args)
                    v_embeds = self.allgather(v_embeds, self.n_gpu, self.args)
                    output = sim_matrix(text_embeds, video_embeds)

                    if self.config["loss"]["type"] == "EgoNCE":
                        sim_v = sim_matrix(v_embeds, v_embeds)
                        sim_n = sim_matrix(n_embeds, n_embeds)
                        loss = self.loss(output, sim_v, sim_n)
                    else:
                        loss = self.loss(output)

                loss.backward()

                self.optimizer.step()

                if self.writer is not None and self.args.rank == 0:
                    # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
                    total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch - 1) * total + current
                    self.writer.add_scalar(
                        f"Loss_training/loss_{dl_idx}",
                        loss.detach().item(),
                        final_total,
                    )

                total_loss[dl_idx] += loss.detach().item()

                # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                if batch_idx % self.log_step == 0 and self.args.rank == 0:
                    self.logger.info(
                        "Train Epoch: {} dl{} {} Loss: {:.6f}".format(
                            epoch,
                            dl_idx,
                            self._progress(batch_idx, dl_idx),
                            loss.detach().item(),
                        )
                    )

                self.optimizer.zero_grad()
            if batch_idx == self.len_epoch:
                break

        log = {
            f"loss_{dl_idx}": total_loss[dl_idx] / self.len_epoch
            for dl_idx in range(len(self.data_loader))
        }

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(
                    f"Loss_training/loss_total_{dl_idx}", tl, epoch - 1
                )

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)

        gt_arr = {x: [] for x in range(len(self.valid_data_loader))}
        pred_arr = {x: [] for x in range(len(self.valid_data_loader))}
        type_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(tqdm(dl)):
                    data["video"] = data["video"][0]  # remove batch
                    data["text"] = data["text"]

                    if self.tokenizer is not None:
                        data["text"] = self.tokenizer(
                            data["text"],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )
                    data["text"] = {
                        key: val.to(self.device) for key, val in data["text"].items()
                    }
                    data["video"] = data["video"].to(self.device)
                    text_embed, vid_embed = self.model(data, return_embeds=True)

                    data_gt = data["correct"][0].to(self.device).unsqueeze(0)
                    data_pred = sim_matrix(text_embed, vid_embed)
                    data_type = data["type"][0].to(self.device).unsqueeze(0)

                    # if isinstance(self.model, nn.DataParallel) and data["video"].shape[0] < len(self.model.device_ids):
                    # Note that if some batch has size smaller than the GPU size, `DataParallel` will fail.
                    # It can happen with the last batch of the dataset, depending on its size.
                    # This avoids using `DataParallel` in this case, and supposes the entire batch fits in one GPU.
                    #    text_embed, vid_embed = self.model.module(data, return_embeds=True)
                    # else:
                    #    text_embed, vid_embed = self.model(data, return_embeds=True)
                    data_gt_all = [torch.zeros_like(data_gt) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_gt_all, data_gt)
                    data_gt_all = torch.cat(data_gt_all, dim=0)

                    data_pred_all = [
                        torch.zeros_like(data_pred) for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(data_pred_all, data_pred)
                    data_pred_all = torch.cat(data_pred_all, dim=0)

                    data_type_all = [
                        torch.zeros_like(data_type) for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(data_type_all, data_type)
                    data_type_all = torch.cat(data_type_all, dim=0)

                    gt_arr[dl_idx].append(data_gt_all.cpu())
                    pred_arr[dl_idx].append(data_pred_all.cpu())
                    type_arr[dl_idx].append(data_type_all.cpu())

            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    self.writer.add_scalar(
                        f"Loss_val/loss_total_{dl_idx}", tl, epoch - 1
                    )

        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            gt_arr_cat = torch.cat(gt_arr[dl_idx])
            pred_arr_cat = torch.cat(pred_arr[dl_idx])
            type_cat = torch.cat(type_arr[dl_idx])

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(pred_arr_cat, gt_arr_cat, type_cat)
                if self.args.rank == 0:
                    self.logger.info(
                        verbose(
                            epoch=epoch,
                            metrics=res,
                            name=self.valid_data_loader[dl_idx].dataset_name,
                        )
                    )
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(
                        res,
                        mode=metric_name,
                        name=self.valid_data_loader[dl_idx].dataset_name,
                    )
                    # for key, val in to_write.items():
                    #     self.writer.log_scalar(key, val)
                    for key, val in to_write.items():
                        key = key.replace("[", "_").replace("]", "_")
                        self.writer.add_scalar(
                            f"Val_metrics_{dl_idx}/{key}", val, epoch - 1
                        )

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {
                f"val_loss_{dl_idx}": total_val_loss[dl_idx]
                / len(self.valid_data_loader[dl_idx])
                for dl_idx in range(len(self.valid_data_loader))
            }
            res_dict["nested_val_metrics"] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader[dl_idx], "n_samples"):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class Multi_Trainer_aud_dist(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from Multi_BaseTrainer_dist.
    """

    def __init__(
        self,
        args,
        model,
        loss,
        metrics,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
        writer=None,
        visualizer=None,
        tokenizer=None,
        tokenizer_v=None,
        max_samples_per_epoch=50000,
        start_epoch=1,
        model_v=None,
    ):
        super().__init__(
            args,
            model,
            loss,
            metrics,
            optimizer,
            config,
            writer,
            start_epoch=start_epoch,
            model_v=model_v,
        )
        self.config = config
        self.args = args
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        if self.valid_data_loader is not None:
            self.len_val_epoch = min([len(x) for x in valid_data_loader])
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer_v = tokenizer_v
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.join_method = None
        self.normalisation = None
        if self.model_v is not None:
            try:
                self.join_method = self.config["trainer"]["join_method"]
                if self.args.rank == 0:
                    self.logger.info(
                        f"join_method is defined in config to be {self.join_method}"
                    )
            except Exception as e:
                if self.args.rank == 0:
                    self.logger.info(
                        "join_method not defined in config so assumed to be simple"
                    )
                self.join_method = "simple"
            try:
                self.vtoa_ratio = self.config["trainer"]["vtoa_ratio"]
                if self.args.rank == 0:
                    self.logger.info(
                        f"vtoa_ratio is defined in config to be {self.vtoa_ratio}:1"
                    )
            except Exception as e:
                if self.args.rank == 0:
                    self.logger.info(
                        "vtoa_ratio not defined in config so assumed to be 1:1"
                    )
                self.vtoa_ratio = 1
            try:
                self.normalisation = self.config["trainer"]["normalisation"]
                if self.args.rank == 0:
                    self.logger.info(
                        f"normalisation is defined in config to be {self.normalisation}"
                    )
            except Exception as e:
                if self.args.rank == 0:
                    self.logger.info(
                        "normalisation not defined in config so assumed to be None"
                    )
        self.use_gpt = (
            self.config["data_loader"][0]["args"]
            .get("aud_params", {"use_gpt": False})
            .get("use_gpt", False)
        )
        self.seed = self.config["trainer"].get("seed", 0)
        if self.args.rank == 0:
            self.logger.info(
                f"the use of gpt is defined in config to be {self.use_gpt} and seed is {self.seed}"
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        self.model.train()
        if self.model_v is not None:
            if self.args.rank == 0 and epoch == 1:
                print("TRAIN: Video model is provided")
            self.model_v.train()
            if self.model_tav is not None:
                if self.args.rank == 0 and epoch == 1:
                    print("TRAIN: Projection model is provided")
                self.model_tav.train()
        else:
            if self.args.rank == 0 and epoch == 1:
                print("TRAIN: Video model is not provided")

        total_loss = [0] * len(self.data_loader)
        total_loss_val = [0] * len(self.valid_data_loader)
        gt_arr = {x: [] for x in range(len(self.valid_data_loader))}
        pred_arr = {x: [] for x in range(len(self.valid_data_loader))}
        type_arr = {x: [] for x in range(len(self.valid_data_loader))}
        criterion_val_2 = EgoNCE(noun=False, verb=False)
        counter_val = -1

        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        val_period = max(
            int(len(*self.data_loader) / len(*self.valid_data_loader)) + 1, 1
        )
        no_val_eg = len(*self.valid_data_loader)
        if self.config["trainer"]["track_train_val"] is True:
            iterator = iter(*self.valid_data_loader)
            if self.args.rank == 0:
                print(
                    f"Evaluating on one validation example every {val_period} train iter"
                )
        if self.args.rank == 0:
            print(f"len(*self.data_loader) is {len(*self.data_loader)}")
            print(f"self.data_loader[0].batch_size is {self.data_loader[0].batch_size}")
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                if self.args.rank == 0:
                    print(
                        f"Got to this if l329 in trainer_egoclip {self.total_batch_sum} and {self.max_samples_per_epoch}"
                    )
                break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if "video_neg" in data.keys():  # w/ negative sampling
                    data["text"] = data["text"] + data["text_neg"]
                    data["video"] = torch.cat(
                        (data["video"], data["video_neg"]), axis=0
                    )

                    data["noun_vec"] = torch.cat(
                        (data["noun_vec"], data["noun_vec_neg"]), axis=0
                    )
                    data["verb_vec"] = torch.cat(
                        (data["verb_vec"], data["verb_vec_neg"]), axis=0
                    )

                if self.tokenizer is not None:
                    data["text"] = self.tokenizer(
                        data["text"], return_tensors="pt", padding=True, truncation=True
                    )
                data["text"] = {
                    key: val.to(self.device) for key, val in data["text"].items()
                }
                data["video"] = data["audio"]
                # data['video'] = data['video'].to(self.device)
                n_embeds = data["noun_vec"].to(self.device)
                v_embeds = data["verb_vec"].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # print("Got to 352 in trainer_egoclip")
                    # self.logger.info("Got to 353 in trainer_egoclip")
                    text_embeds, video_embeds = self.model(
                        data["video"],
                        data["text"],
                        device=self.device,
                        return_embeds=True,
                    )
                    video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
                    text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)
                    n_embeds = self.allgather(n_embeds, self.n_gpu, self.args)
                    v_embeds = self.allgather(v_embeds, self.n_gpu, self.args)
                    output = sim_matrix(text_embeds, video_embeds)

                    if self.config["loss"]["type"] == "EgoNCE":
                        sim_v = sim_matrix(v_embeds, v_embeds)
                        sim_n = sim_matrix(n_embeds, n_embeds)
                        loss = self.loss(output, sim_v, sim_n)
                    else:
                        loss = self.loss(output)

                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                if self.writer is not None and self.args.rank == 0:
                    # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
                    total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch - 1) * total + current
                    self.writer.add_scalar(
                        f"Loss_training/loss_{dl_idx}",
                        loss.detach().item(),
                        final_total,
                    )
                    self.writer.add_scalar(
                        f"Learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        final_total,
                    )

                total_loss[dl_idx] += loss.detach().item()

                # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                if batch_idx % self.log_step == 0 and self.args.rank == 0:
                    self.logger.info(
                        "[{}] Train Epoch: {} dl{} {} Loss: {:.6f}".format(
                            datetime.now().strftime(r"%m%d_%H:%M:%S"),
                            epoch,
                            dl_idx,
                            self._progress(batch_idx, dl_idx),
                            loss.detach().item(),
                        )
                    )

                self.optimizer.zero_grad()
            # self.logger.info(f'Got to line 393')

            ####### This only checks what is happening with the validation data during training
            if (
                self.config["trainer"]["track_train_val"] is True
                and batch_idx % val_period == 0
            ):
                try:
                    data_val = next(iterator)
                except Exception as e:
                    print(
                        f"Exception {e} reached, probably no next item in val data_loader"
                    )
                    continue
                    # then assume we must tokenize the input, e.g. its a string
                if "video_neg" in data_val.keys():  # w/ negative sampling
                    data_val["text"] = data_val["text"] + data_val["text_neg"]
                    data_val["video"] = torch.cat(
                        (data_val["video"], data_val["video_neg"]), axis=0
                    )

                    data_val["noun_vec"] = torch.cat(
                        (data_val["noun_vec"], data_val["noun_vec_neg"]), axis=0
                    )
                    data_val["verb_vec"] = torch.cat(
                        (data_val["verb_vec"], data_val["verb_vec_neg"]), axis=0
                    )

                if self.tokenizer is not None:
                    data_val["text"] = self.tokenizer(
                        data_val["text"],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                data_val["text"] = {
                    key: val.to(self.device) for key, val in data_val["text"].items()
                }
                data_val["video"] = data_val["audio"]
                # data_val['video'] = data_val['video'].to(self.device)
                n_embeds_val = data_val["noun_vec"].to(self.device)
                v_embeds_val = data_val["verb_vec"].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # print("Got to 352 in trainer_egoclip")
                    # self.logger.info("Got to 353 in trainer_egoclip")
                    text_embeds_val, video_embeds_val = self.model(
                        data_val["video"],
                        data_val["text"],
                        device=self.device,
                        return_embeds=True,
                    )
                    video_embeds_val = self.allgather(
                        video_embeds_val, self.n_gpu, self.args
                    )
                    text_embeds_val = self.allgather(
                        text_embeds_val, self.n_gpu, self.args
                    )
                    n_embeds_val = self.allgather(n_embeds_val, self.n_gpu, self.args)
                    v_embeds_val = self.allgather(v_embeds_val, self.n_gpu, self.args)
                    output_val = sim_matrix(text_embeds_val, video_embeds_val)

                    if self.config["loss"]["type"] == "EgoNCE":
                        sim_v_val = sim_matrix(v_embeds_val, v_embeds_val)
                        sim_n_val = sim_matrix(n_embeds_val, n_embeds_val)
                        loss_val = self.loss(output_val, sim_v_val, sim_n_val)
                    else:
                        loss_val = self.loss(output_val)
                    try:
                        total_loss_val[0] += loss_val.detach().item()
                    except Exception as e:
                        print(
                            f"Error {e} at batch_idx {batch_idx} with val_period {val_period} and ratio {batch_idx//val_period}"
                        )
                        raise IndexError

                if self.writer is not None and self.args.rank == 0:
                    self.writer.add_scalar(
                        f"Loss_val_during_training/loss_val_0",
                        loss_val.detach().item(),
                        batch_idx // val_period,
                    )
            ##############

            if batch_idx == self.len_epoch:
                self.logger.info(
                    f"Got to line 395 and batch_idx is {batch_idx} and len_Epoch is {self.len_epoch}"
                )
                break

        log = {
            f"loss_{dl_idx}": total_loss[dl_idx] / self.len_epoch
            for dl_idx in range(len(self.data_loader))
        }

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(
                    f"Loss_training/loss_total_{dl_idx}", tl, epoch - 1
                )
            if self.config["trainer"]["track_train_val"] is True:
                for dl_idx_val in range(len(self.valid_data_loader)):
                    tl_val = total_loss_val[dl_idx_val] / self.len_val_epoch
                    self.writer.add_scalar(
                        f"Loss_val_during_training/loss_val_total_{dl_idx_val}",
                        tl_val,
                        epoch - 1,
                    )
        if self.args.rank == 0:
            self.logger.info(
                f"Now epoch is {epoch} and if it is divisible by 2 then validation step"
            )
            self._save_checkpoint(epoch, save_best=False)
        if (
            epoch % 1 == 0
            and epoch != 0
            and self.config["trainer"]["track_train_val"] is False
        ):
            if self.do_validation:
                val_log = self._valid_epoch(epoch)
                if self.args.rank == 0:
                    log.update(val_log)
        if self.lr_scheduler is None:
            self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    # @profile
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        data_type_dict = {1: "inter", 2: "intra"}
        self.model.eval()
        if self.model_v is not None:
            if self.args.rank == 0 and epoch == 1:
                print("VALID: Video model is provided")
            self.model_v.eval()
            if self.model_tav is not None:
                self.model_tav.eval()
        else:
            if self.args.rank == 0 and epoch == 1:
                print("VALID: Video model is not provided")
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_loss_2 = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)

        criterion_val = MaxMarginRankingLoss()
        # criterion_val_2 = EgoMILNCE()
        criterion_val_2 = EgoNCE(noun=False, verb=False)

        gt_arr = {x: [] for x in range(len(self.valid_data_loader))}
        pred_arr = {x: [] for x in range(len(self.valid_data_loader))}
        type_arr = {x: [] for x in range(len(self.valid_data_loader))}
        total = len(self.valid_data_loader) * self.batch_size
        data_key = "text_gpt" if self.use_gpt else "text"
        print(f"data_key used is {data_key}")
        model_used = "audio" if "arch_vid" not in self.config._config else "both"
        text_doc = ""
        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length

            for dl_idx, dl in enumerate(self.valid_data_loader):
                len_dataloader = len(dl)
                for batch_idx, data in tqdm(enumerate(dl), total=len_dataloader):
                    data_text_orig = data[data_key].copy()
                    data_text_gpt = data["text_gpt"].copy()
                    data_text_official = data["text"].copy()
                    if self.config["arch"]["type"] == "ASE":
                        data_text_a = self.tokenizer(
                            data[data_key],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )
                    else:
                        data_text_a = self.tokenizer(
                            data[data_key],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )

                    if model_used == "both":
                        data["text"] = tokenize_and_move_to_gpu(
                            data["text"],
                            self.tokenizer_v,
                            "video",
                            self.config["arch"]["type"],
                        )
                        self.model_v.eval()

                        data["video"] = data["video"][0]  # remove batch
                        data = move_video_data_to_device(data, self.device)

                        text_embed, vid_embed = self.model_v(data, return_embeds=True)
                        data_pred = sim_matrix(
                            text_embed, vid_embed, norm=self.normalisation
                        )

                    if self.config["arch"]["type"] == "CLAP":
                        text_embed_aud, aud_embed = self.model(
                            data["audio"],
                            data_text_a,
                            device=self.device,
                            return_embeds=True,
                        )
                    elif self.config["arch"]["type"] == "ASE":
                        # # if want to send in as batch of 1
                        # data["audio"]["waveform"] = data["audio"]["waveform"].to(
                        #     self.device
                        # )
                        # if want to send in as batch of 5
                        data["audio"]["waveform"] = data["audio"]["waveform"][0].to(
                            self.device
                        )
                        aud_embed = self.model.encode_audio(data["audio"]["waveform"])
                        text_embed_aud = self.model.encode_text(data_text_orig)

                    data_gt = data["correct"][0].to(self.device).unsqueeze(0)

                    # mask = []
                    # batch_size = text_embed_aud.shape[0]
                    # for i in range(0, batch_size):
                    #     mask.append(torch.tensor([0, 1, 2, 3, 4])*batch_size + i)
                    # final_mask = torch.cat(mask, dim=0)
                    # aud_embed = aud_embed[final_mask, :]

                    # This is commented v just to allow the model to use video only
                    if self.args.rank == 0 and batch_idx == 0:
                        print(f"using normalisation {self.normalisation}")
                    data_pred_aud = sim_matrix(
                        text_embed_aud, aud_embed, norm=self.normalisation
                    )
                    if self.model_v is not None:
                        if self.args.rank == 0 and batch_idx == 0:
                            print(f"using vtoa ratio {self.vtoa_ratio}")
                        data_pred = (self.vtoa_ratio * data_pred + data_pred_aud) / (
                            self.vtoa_ratio + 1
                        )
                    else:
                        data_pred = data_pred_aud.detach()
                    augmented_data_pred = torch.eye(5)
                    augmented_data_pred[data["correct"][0]] = data_pred
                    loss = criterion_val(augmented_data_pred)
                    # loss_2 = criterion_val_2(augmented_data_pred.to(self.device), 5)
                    loss_2 = criterion_val_2(
                        augmented_data_pred.to(self.device), None, None
                    )
                    total_val_loss[dl_idx] += loss.detach().item()
                    total_val_loss_2[dl_idx] += loss_2.detach().item()
                    # self.logger.info(f'Got similarity matrix')
                    data_type = data["type"][0].to(self.device).unsqueeze(0)
                    if data_pred.argmax().item() != data_gt:
                        text_doc += f"data_key is {data_key}, orig descr {data_text_official} and gpt descr is {data_text_gpt} and data_type is {data_type_dict[data_type.item()]}\n"
                    # print(f'Data type is {data_type}')

                    # if isinstance(self.model, nn.DataParallel) and data["video"].shape[0] < len(self.model.device_ids):
                    # Note that if some batch has size smaller than the GPU size, `DataParallel` will fail.
                    # It can happen with the last batch of the dataset, depending on its size.
                    # This avoids using `DataParallel` in this case, and supposes the entire batch fits in one GPU.
                    #    text_embed, vid_embed = self.model.module(data, return_embeds=True)
                    # else:
                    #    text_embed, vid_embed = self.model(data, return_embeds=True)
                    data_gt_all = [torch.zeros_like(data_gt) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_gt_all, data_gt)
                    data_gt_all = torch.cat(data_gt_all, dim=0)

                    data_pred_all = [
                        torch.zeros_like(data_pred) for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(data_pred_all, data_pred)
                    data_pred_all = torch.cat(data_pred_all, dim=0)

                    data_type_all = [
                        torch.zeros_like(data_type) for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(data_type_all, data_type)
                    data_type_all = torch.cat(data_type_all, dim=0)

                    gt_arr[dl_idx].append(data_gt_all.cpu())
                    pred_arr[dl_idx].append(data_pred_all.cpu())
                    type_arr[dl_idx].append(data_type_all.cpu())

            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    print(
                        f"Now looping through dataloader and saving validation loss. Total val loss of dl_idx is {total_val_loss[dl_idx]}"
                    )
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    tl_2 = total_val_loss_2[dl_idx] / len(
                        self.valid_data_loader[dl_idx]
                    )
                    self.writer.add_scalar(
                        f"Loss_val/loss_total_{dl_idx}", tl, epoch - 1
                    )
                    self.writer.add_scalar(
                        f"Loss_val_2/loss_total_{dl_idx}", tl_2, epoch - 1
                    )
        with open(
            f"/scratch/shared/beegfs/oncescu/coding/libs/pt/egovlp-copy/egovlp/{data_key}_full_test.txt",
            "w",
        ) as f:
            # saving here the text descriptions that were wrongly classified
            f.write(text_doc)
        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}
            gt_arr_cat = torch.cat(gt_arr[dl_idx])
            pred_arr_cat = torch.cat(pred_arr[dl_idx])
            type_cat = torch.cat(type_arr[dl_idx])
            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(pred_arr_cat, gt_arr_cat, type_cat)
                if self.args.rank == 0:
                    self.logger.info(
                        verbose(
                            epoch=epoch,
                            metrics=res,
                            name=self.valid_data_loader[dl_idx].dataset_name,
                        )
                    )
                    save_results(
                        config=self.config,
                        args=self.args,
                        res=res,
                        model_used=model_used,
                        metric_name=metric_name,
                        use_gpt=self.use_gpt,
                        test_file=self.valid_data_loader[dl_idx].dataset.test_file,
                        seed=self.seed,
                        right_sec=self.valid_data_loader[dl_idx].dataset.right_sec,
                        left_sec=self.valid_data_loader[dl_idx].dataset.left_sec,
                    )
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(
                        res,
                        mode=metric_name,
                        name=self.valid_data_loader[dl_idx].dataset_name,
                    )
                    # for key, val in to_write.items():
                    #     self.writer.log_scalar(key, val)
                    for key, val in to_write.items():
                        key = key.replace("[", "_").replace("]", "_")
                        self.writer.add_scalar(
                            f"Val_metrics_{dl_idx}/{key}", val, epoch - 1
                        )

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {
                f"val_loss_{dl_idx}": total_val_loss_2[dl_idx]
                / len(self.valid_data_loader[dl_idx])
                for dl_idx in range(len(self.valid_data_loader))
            }
            # print(res_dict)
            # res_dict = {f'val_loss_2_{dl_idx}': total_val_loss_2[dl_idx] / len(self.valid_data_loader[dl_idx])
            #             for dl_idx in range(len(self.valid_data_loader))}
            print(res_dict)
            res_dict["nested_val_metrics"] = nested_metrics

        return res_dict

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            # if self.writer is not None:
            #     self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
        if self.args.rank == 0:
            print("[INFO] Learning rate for next epoch is: {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _progress(self, batch_idx, dl_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader[dl_idx], "n_samples"):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, name="TEST"):
    msg = ""
    for key in metrics.keys():
        acc = metrics[key]
        msg += f"{name:s} epoch {epoch}, {key:s}, Acc: {acc:.1f};    "
    print(msg)
    return msg


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
