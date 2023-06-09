from __future__ import division, print_function

import argparse
import os
import time
from functools import partial

import core.datasets as datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import evaluate
import viz
from core.metrics import compute_epe,  merge_metrics, process_metrics

from core.flowformer import build_flowformer
from core.deq.arg_utils import add_deq_args
from core.utils.flow_viz import flow_to_image

from torch.cuda.amp import GradScaler

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fixed_point_correction(
    flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, cal_epe=True
):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    if cal_epe:
        epe = compute_epe(flow_preds[-1], flow_gt, valid)
        return flow_loss, epe
    else:
        return flow_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wdecay,
        eps=args.epsilon
    )

    if args.schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.num_steps,
            eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            args.num_steps+100,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='linear'
        )

    return optimizer, scheduler


class Logger:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.total_steps = args.resume_iter if args.resume_iter > 0 else 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        sorted_keys = sorted(self.running_loss.keys())
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted_keys]
        training_str = f"[Step {self.total_steps+1:6d}, lr {self.scheduler.get_last_lr()[0]:.7}]   "
        metrics_str = ", ".join([f"{name}:{val:10.4f}" for (
            name, val) in zip(sorted_keys, metrics_data)])

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter("runs/" + args.name_per_run)

        for k in self.running_loss:
            self.writer.add_scalar(
                k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def write_img(self, img, img_name, step):
        if self.writer is None:
            self.writer = SummaryWriter()

        self.writer.add_image(img_name, img, step)

    def close(self):
        self.writer.close()


def visualize_validation_results(model, data_blob, logger, img_name, steps, args):
    model.eval()
    with torch.no_grad():
        imgs, _, _ = data_blob
        image1 = imgs[:, 0, ...]
        image2 = imgs[:, 1, ...]
        image1 = image1.to(DEVICE)
        image2 = image2.to(DEVICE)

        with autocast(enabled=args.mixed_precision):
            _, _, info = model(image1, image2)
        flow_predictions = info.get("flow_predictions", [])

    model.train()
    flow_predictions = [
        el.clone().detach().cpu().numpy()[0, ...] for el in flow_predictions
    ]

    flow_predictions = flow_predictions[:1] + flow_predictions[-2:]

    flow_prediction_line = np.concatenate(flow_predictions, axis=2)
    try:
        flow_prediction_line = flow_to_image(flow_prediction_line.T).T
        logger.write_img(flow_prediction_line, img_name, steps)
    except Exception as ex:
        print(ex)


def train(cfg, args):
    stats = dict()
    for i in range(args.start_run, args.total_run+1):
        if args.restore_name is not None:
            args.restore_name_per_run = 'checkpoints/' + \
                args.restore_name + f'-run-{i}.pth'
        args.name_per_run = args.name + f'-run-{i}'
        best_chairs, best_sintel, best_kitti = train_once(cfg, args)

        if best_chairs['epe'] < 100:
            stats['chairs'] = stats.get('chairs', []) + [best_chairs['epe']]
        if best_sintel['clean-epe'] < 100:
            stats['sintel clean'] = stats.get(
                'sintel clean', []) + [best_sintel['clean-epe']]
            stats['sintel final'] = stats.get(
                'sintel final', []) + [best_sintel['final-epe']]
        if best_kitti['epe'] < 100:
            stats['kitti epe'] = stats.get(
                'kitti epe', []) + [best_kitti['epe']]
            stats['kitti f1'] = stats.get('kitti f1', []) + [best_kitti['f1']]

        write_stats(args, stats)

        # reset resume iters
        args.resume_iter = -1


def write_stats(args, stats):
    log_path = f'stats/{args.name}_{args.stage}_total_{args.total_run}_start_{args.start_run}.txt'
    with open(log_path, 'w+') as f:
        for key, values in stats.items():
            f.write(f'{key}: {values}\n')


def train_once(cfg, args):
    flowformer = build_flowformer(cfg, args)
    model = nn.DataParallel(flowformer, device_ids=args.gpus)
    print(f"Parameter Count: {count_parameters(model):.3}M")

    if args.restore_name is not None:
        model.load_state_dict(torch.load(
            args.restore_name_per_run), strict=False)
        print(f'Load from {args.restore_name_per_run}')

    if args.resume_iter > 0:
        restore_path = f'checkpoints/{args.resume_iter}_{args.name_per_run}.pth'
        model.load_state_dict(torch.load(restore_path), strict=False)
        print(f'Resume from {restore_path}')

    model.to(DEVICE)
    model.train()

    # if args.stage != 'chairs' and not args.active_bn:
    #     model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)

    aug_params = {
        'crop_size': args.image_size,
    }
    val_data = datasets.MpiSintel(
        aug_params,
        split='training',
        dstype="clean",
        seq_len=2,
    )
    val_data_blob = [el.unsqueeze(0) for el in val_data[0]]

    optimizer, scheduler = fetch_optimizer(args, model)
    scheduler.last_epoch = args.resume_iter if args.resume_iter > 0 else -1

    total_steps = args.resume_iter if args.resume_iter > 0 else 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(scheduler)

    add_noise = True
    best_chairs = {"epe": 1e8}
    best_sintel = {"clean-epe": 1e8, "final-epe": 1e8}
    best_kitti = {"epe": 1e8, "f1": 1e8}
    should_keep_training = True
    while should_keep_training:

        timer = 0

        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            imgs, flows, valids = data_blob

            for j in range(imgs.shape[1]-1):
                optimizer.zero_grad()
                image1 = imgs[:, j, ...]
                image2 = imgs[:, j+1, ...]
                flow = flows[:, j, ...]
                valid = valids[:, j, ...]

                image1 = image1.to(DEVICE)
                image2 = image2.to(DEVICE)
                flow = flow.to(DEVICE)
                valid = valid.to(DEVICE)

                if args.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (
                        image1 + stdv * torch.randn(*image1.shape).to(DEVICE)
                    ).clamp(0.0, 255.0)
                    image2 = (
                        image2 + stdv * torch.randn(*image2.shape).to(DEVICE)
                    ).clamp(0.0, 255.0)

                start_time = time.time()

                fc_loss = partial(fixed_point_correction, gamma=args.gamma)

                # TODO extranct the flow_init, net, corr, etc
                flow_predictions, info = model(image1, image2)
                flow_loss, epe = fc_loss(flow_predictions, flow, valid)

                batch_metrics = process_metrics(epe, info)

                metrics = merge_metrics(batch_metrics)
                scaler.scale(flow_loss.mean()).backward()

                end_time = time.time()
                timer += end_time - start_time

                scaler.unscale_(optimizer)
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                logger.push(metrics)

            if (total_steps + 1) % args.time_interval == 0:
                print(f'Exp {args.name_per_run} Average Time: {timer / args.time_interval}')
                timer = 0

            if (total_steps + 1) % args.save_interval == 0:
                PATH = f'checkpoints/{total_steps+1}_{args.name_per_run}.pth'
                torch.save(model.state_dict(), PATH)

            if total_steps % args.eval_interval == args.eval_interval - 1:
                results = {}

                visualize_validation_results(
                    model,
                    val_data_blob,
                    logger,
                    "train-sintel",
                    total_steps,
                    args,
                )
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        res = evaluate.validate_chairs(
                            model.module,
                            mixed_precision=args.mixed_precision,
                            sradius_mode=args.sradius_mode,
                            best=best_chairs
                        )
                        best_chairs['epe'] = min(
                            res['chairs'], best_chairs['epe']
                        )
                        results.update(res)
                    elif val_dataset == 'things':
                        results.update(
                            evaluate.validate_things(
                                model.module,
                                mixed_precision=args.mixed_precision,
                                sradius_mode=args.sradius_mode
                            )
                        )
                    elif val_dataset == 'sintel':
                        res = evaluate.validate_sintel(
                            model.module,
                            mixed_precision=args.mixed_precision,
                            sradius_mode=args.sradius_mode,
                            best=best_sintel
                        )
                        best_sintel['clean-epe'] = min(
                            res['clean'], best_sintel['clean-epe']
                        )
                        best_sintel['final-epe'] = min(
                            res['final'], best_sintel['final-epe']
                        )
                        results.update(res)
                    elif val_dataset == 'kitti':
                        res = evaluate.validate_kitti(
                            model.module,
                            mixed_precision=args.mixed_precision,
                            sradius_mode=args.sradius_mode,
                            best=best_kitti
                        )
                        best_kitti['epe'] = min(
                            res['kitti-epe'], best_kitti['epe'])
                        best_kitti['f1'] = min(
                            res['kitti-f1'], best_kitti['f1'])
                        results.update(res)

                logger.write_dict(results)

                model.train()
                # if args.stage != 'chairs':
                #     model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = f'checkpoints/{args.name_per_run}.pth'
    torch.save(model.state_dict(), PATH)

    return best_chairs, best_sintel, best_kitti


def val(cfg, args):
    flowformer = build_flowformer(args)
    model = nn.DataParallel(flowformer, device_ids=args.gpus)
    print(f"Parameter Count: {count_parameters(model):.3}M")

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print(f'Load from {args.restore_ckpt}')

    model.to(DEVICE)
    model.eval()

    for val_dataset in args.validation:
        if val_dataset == 'chairs':
            evaluate.validate_chairs(
                model.module,
                mixed_precision=args.mixed_precision,
                sradius_mode=args.sradius_mode
            )
        elif val_dataset == 'things':
            evaluate.validate_things(
                model.module,
                mixed_precision=args.mixed_precision,
                sradius_mode=args.sradius_mode
            )
        elif val_dataset == 'sintel':
            evaluate.validate_sintel(
                model.module,
                mixed_precision=args.mixed_precision,
                sradius_mode=args.sradius_mode
            )
        elif val_dataset == 'kitti':
            evaluate.validate_kitti(
                model.module,
                mixed_precision=args.mixed_precision,
                sradius_mode=args.sradius_mode
            )


def test(cfg, args):
    flowformer = build_flowformer(cfg, args)
    model = nn.DataParallel(flowformer, device_ids=args.gpus)
    print(f"Parameter Count: {count_parameters(model):.3}M")

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.to(DEVICE)
    model.eval()

    for test_dataset in args.test_set:
        if test_dataset == 'sintel':
            evaluate.create_sintel_submission(
                model.module,
                mixed_precision=args.mixed_precision,
                output_path=args.output_path,
                fixed_point_reuse=args.fixed_point_reuse,
                warm_start=args.warm_start
            )
        elif test_dataset == 'kitti':
            evaluate.create_kitti_submission(
                model.module,
                mixed_precision=args.mixed_precision,
                output_path=args.output_path
            )


def visualize(cfg, args):
    flowformer = build_flowformer(cfg, args)
    model = nn.DataParallel(flowformer, device_ids=args.gpus)
    print(f"Parameter Count: {count_parameters(model):.3}M")

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.to(DEVICE)
    model.eval()

    for viz_dataset in args.viz_set:
        for split in args.viz_split:
            if viz_dataset == 'sintel':
                viz.sintel_visualization(
                    model.module,
                    split=split,
                    output_path=args.output_path,
                    fixed_point_reuse=args.fixed_point_reuse,
                    warm_start=args.warm_start
                )
            elif viz_dataset == 'kitti':
                viz.kitti_visualization(
                    model.module,
                    split=split,
                    output_path=args.output_path
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true',
                        help="Enable Eval mode.")
    parser.add_argument('--test', action='store_true',
                        help="Enable Test mode.")
    parser.add_argument('--viz', action='store_true', help="Enable Viz mode.")
    parser.add_argument('--fixed_point_reuse',
                        action='store_true', help="Enable fixed point reuse.")
    parser.add_argument('--warm_start', action='store_true',
                        help="Enable warm start.")

    parser.add_argument('--name', default='deq-flow',
                        help="name your experiment")
    parser.add_argument(
        '--stage', help="determines which dataset to use for training")

    parser.add_argument('--total_run', type=int, default=1,
                        help="total number of runs")
    parser.add_argument('--start_run', type=int, default=1,
                        help="begin from the given number of runs")
    parser.add_argument('--restore_name', help="restore experiment name")
    parser.add_argument('--resume_iter', type=int, default=-1,
                        help="resume from the given iterations")

    parser.add_argument('--tiny', action='store_true',
                        help='use a tiny model for ablation study')
    parser.add_argument('--large', action='store_true',
                        help='use a large model')
    parser.add_argument('--huge', action='store_true', help='use a huge model')
    parser.add_argument('--gigantic', action='store_true',
                        help='use a gigantic model')
    parser.add_argument('--old_version', action='store_true',
                        help='use the old design for flow head')

    parser.add_argument(
        '--restore_ckpt', help="restore checkpoint for val/test/viz")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--test_set', type=str, nargs='+')
    parser.add_argument('--viz_set', type=str, nargs='+')
    parser.add_argument('--viz_split', type=str, nargs='+', default=['test'])
    parser.add_argument('--output_path', help="output path for evaluation")

    parser.add_argument('--eval_interval', type=int,
                        default=5000, help="evaluation interval")
    parser.add_argument('--save_interval', type=int,
                        default=5000, help="saving interval")
    parser.add_argument('--time_interval', type=int,
                        default=500, help="timing interval")

    parser.add_argument('--gma', action='store_true', help='use gma')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int,
                        nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--schedule', type=str,
                        default="onecycle", help="learning rate schedule")
    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--vdropout', type=float, default=0.0,
                        help="variational dropout added to BasicMotionEncoder for DEQs")
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--active_bn', action='store_true')
    parser.add_argument('--all_grad', action='store_true',
                        help="Remove the gradient mask within DEQ func.")

    # Add args for utilizing DEQ
    add_deq_args(parser)
    args = parser.parse_args()

    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'autoflow':
        from configs.autoflow import get_cfg

    cfg = get_cfg()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.eval:
        val(cfg, args)
    elif args.test:
        test(cfg, args)
    elif args.viz:
        visualize(cfg, args)
    else:
        train(cfg, args)
