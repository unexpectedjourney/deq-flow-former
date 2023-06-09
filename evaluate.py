import os
import time

import core.datasets as datasets
import numpy as np
import torch

from core.utils import frame_utils
from core.utils.utils import InputPadder, forward_interpolate

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


MAX_FLOW = 400

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def create_sintel_submission(
    model, warm_start=False, fixed_point_reuse=False,
    mixed_precision=False, output_path='sintel_submission', **kwargs
):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    seq_len = 2
    for dstype in ['clean', 'final']:
        jump_margin = 0
        test_dataset = datasets.MpiSintel(
            split='test', aug_params=None, seq_len=seq_len, dstype=dstype)

        sequence_prev, flow_prev, fixed_point = None, None, None
        for test_id in range(len(test_dataset)):
            inner_test_id = test_id + jump_margin
            if inner_test_id >= len(test_dataset):
                break
            imgs, (sequence, frame) = test_dataset[test_id + jump_margin]
            if sequence != sequence_prev:
                flow_prev = None
                fixed_point = None

            for j in range(imgs.shape[0] - 1):
                if j:
                    jump_margin += 1
                image1 = imgs[j, ...]
                image2 = imgs[j+1, ...]

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(
                    image1[None].to(DEVICE),
                    image2[None].to(DEVICE)
                )

                with autocast(enabled=mixed_precision):
                    flow_low, flow_pr, info = model(
                        image1,
                        image2,
                        flow_init=flow_prev,
                        cached_result=fixed_point,
                        **kwargs
                    )
                flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

                # You may choose to use some hacks here,
                # for example, warm start, i.e., reusing the f* part with a borderline check (forward_interpolate),
                # which was orignally taken by RAFT.
                # This trick usually (only) improves the optical flow estimation on the ``ambush_1'' sequence,
                # in terms of clearer background estimation.
                if warm_start:
                    flow_prev = forward_interpolate(flow_low[0])[None].to(DEVICE)

                # Note that the fixed point reuse usually does not improve performance.
                # It facilitates the convergence.
                # To improve performance, the borderline check like ``forward_interpolate'' is necessary.
                if fixed_point_reuse:
                    net, flow_pred_low = info['cached_result']
                    flow_pred_low = forward_interpolate(
                        flow_pred_low[0]
                    )[None].to(DEVICE)
                    fixed_point = (net, flow_pred_low)

                output_dir = os.path.join(output_path, dstype, sequence)
                output_file = os.path.join(output_dir, f'frame{(frame+1+j):04d}.flo')

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                frame_utils.writeFlow(output_file, flow)
                sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(
        model, output_path='kitti_submission', mixed_precision=False
):
    """ Create submission for the KITTI leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        imgs, (frame_id, ) = test_dataset[test_id]
        image1 = imgs[0, ...]
        image2 = imgs[1, ...]

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(
            image1[None].to(DEVICE), image2[None].to(DEVICE))

        with autocast(enabled=mixed_precision):
            _, flow_pr, _ = model(image1, image2)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, mixed_precision=False, **kwargs):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    rho_list = []
    best = kwargs.get("best", {"epe": 1e8})

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        imgs, flow_gt, _ = val_dataset[val_id]
        image1 = imgs[0, ...]
        image2 = imgs[1, ...]
        image1 = image1[None].to(DEVICE)
        image2 = image2[None].to(DEVICE)

        with autocast(enabled=mixed_precision):
            _, flow_pr, info = model(image1, image2, **kwargs)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())
        rho_list.append(info['sradius'].mean().item())

    epe = np.mean(np.concatenate(epe_list))
    best['epe'] = min(epe, best['epe'])
    print(f"Validation Chairs EPE: {epe:.3f} ({best['epe']:.3f})")

    if np.mean(rho_list) != 0:
        print(f"Spectral radius: {np.mean(rho_list):.2f}")

    return {'chairs': epe}


@torch.no_grad()
def validate_things(model, mixed_precision=False, **kwargs):
    """ Peform validation using the FlyingThings3D (test) split """
    model.eval()
    results = {}
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        val_dataset = datasets.FlyingThings3D(split='test', dstype=dstype)
        epe_list = []
        epe_w_mask_list = []
        rho_list = []

        print(f'{dstype} length', len(val_dataset))

        for val_id in range(len(val_dataset)):
            imgs, flow_gts, valids = val_dataset[val_id]
            image1 = imgs[0, ...]
            image2 = imgs[1, ...]
            flow_gt = flow_gts[0]
            valid = valids[0]

            image1 = image1[None].to(DEVICE)
            image2 = image2[None].to(DEVICE)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            with autocast(enabled=mixed_precision):
                flow_low, flow_pr, info = model(image1, image2, **kwargs)
            flow = padder.unpad(flow_pr[0]).cpu()

            # exlude invalid pixels and extremely large diplacements
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            valid = (valid >= 0.5) & (mag < MAX_FLOW)

            loss = (flow - flow_gt)**2

            if torch.any(torch.isnan(loss)):
                print(f'Bad prediction, {val_id}')

            loss_w_mask = valid[None, :] * loss

            if torch.any(torch.isnan(loss_w_mask)):
                print(f'Bad prediction after mask, {val_id}')
                print('Bad pixels num', torch.isnan(loss).sum())
                print(
                    'Bad pixels num after mask',
                    torch.isnan(loss_w_mask).sum()
                )
                continue

            epe = torch.sum(loss, dim=0).sqrt()
            epe_w_mask = torch.sum(loss_w_mask, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            epe_w_mask_list.append(epe_w_mask.view(-1).numpy())
            rho_list.append(info['sradius'].mean().item())

            if (val_id + 1) % 100 == 0:
                print(
                    'EPE', np.mean(epe_list),
                    'EPE w/ mask', np.mean(epe_w_mask_list)
                )

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1) * 100
        px3 = np.mean(epe_all < 3) * 100
        px5 = np.mean(epe_all < 5) * 100

        epe_all_w_mask = np.concatenate(epe_w_mask_list)
        epe_w_mask = np.mean(epe_all_w_mask)
        px1_w_mask = np.mean(epe_all_w_mask < 1) * 100
        px3_w_mask = np.mean(epe_all_w_mask < 3) * 100
        px5_w_mask = np.mean(epe_all_w_mask < 5) * 100

        print("Validation         (%s) EPE: %.3f, 1px: %.2f, 3px: %.2f, 5px: %.2f" % (
            dstype, epe, px1, px3, px5))
        print("Validation w/ mask (%s) EPE: %.3f, 1px: %.2f, 3px: %.2f, 5px: %.2f" %
              (dstype, epe_w_mask, px1_w_mask, px3_w_mask, px5_w_mask))
        results[dstype] = np.mean(epe_list)
        results[dstype+'_w_mask'] = np.mean(epe_w_mask_list)

        if np.mean(rho_list) != 0:
            print(f"Spectral radius ({dstype}): {np.mean(rho_list)}")

    return results


@torch.no_grad()
def validate_sintel(model, mixed_precision=False, **kwargs):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    best = kwargs.get("best", {"clean-epe": 1e8, "final-epe": 1e8})
    results = {}
    seq_len = 2
    for dstype in ['clean', 'final']:
        used_time = []
        used_iters = []
        jump_margin = 0
        val_dataset = datasets.MpiSintel(
            split='training',
            seq_len=seq_len,
            dstype=dstype
        )
        epe_list = []
        rho_list = []
        info = {"sradius": None, "cached_result": None}

        for val_id in range(len(val_dataset)):
            inner_val_id = val_id + jump_margin
            if inner_val_id >= len(val_dataset):
                break
            imgs, flow_gts, _ = val_dataset[inner_val_id]

            cached_result = None
            for j in range(imgs.shape[0] - 1):
                if j:
                    jump_margin += 1
                    net, flow_pred_low = info['cached_result']
                    flow_pred_low = forward_interpolate(
                        flow_pred_low[0]
                    )[None].to(DEVICE)
                    cached_result = (net, flow_pred_low)

                image1 = imgs[j, ...]
                image2 = imgs[j+1, ...]
                flow_gt = flow_gts[j, ...]

                image1 = image1[None].to(DEVICE)
                image2 = image2[None].to(DEVICE)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                with autocast(enabled=mixed_precision):
                    start_point = time.time()
                    flow_low, flow_pr, info = model(
                        image1,
                        image2,
                        cached_result=cached_result,
                        **kwargs
                    )
                    used_time.append(time.time() - start_point)
                    used_iters.append(info["nstep"])
                flow = padder.unpad(flow_pr[0]).cpu()

                epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
                epe_list.append(epe.view(-1).numpy())
                rho_list.append(info['sradius'].mean().item())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1) * 100
        px3 = np.mean(epe_all < 3) * 100
        px5 = np.mean(epe_all < 5) * 100

        best[dstype+'-epe'] = min(epe, best[dstype+'-epe'])
        print(f"({dstype}-test) Mean update time value: {np.mean(used_time)}")
        print(f"({dstype}-test) Mean update iters value: {np.mean(used_iters)}")

        print(f"Validation ({dstype}) EPE: {epe:.3f} ({best[dstype+'-epe']:.3f}), 1px: {px1:.2f}, 3px: {px3:.2f}, 5px: {px5:.2f}")
        results[dstype] = np.mean(epe_list)

        if np.mean(rho_list) != 0:
            print(f"Spectral radius ({dstype}): {np.mean(rho_list)}")

    return results


@torch.no_grad()
def validate_kitti(model, mixed_precision=False, **kwargs):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    best = kwargs.get("best", {"epe": 1e8, "f1": 1e8})
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list, rho_list = [], [], []
    for val_id in range(len(val_dataset)):
        imgs, flow_gts, valid_gts = val_dataset[val_id]
        image1 = imgs[0, ...]
        image2 = imgs[1, ...]
        flow_gt = flow_gts[0]
        valid_gt = valid_gts[0]

        image1 = image1[None].to(DEVICE)
        image2 = image2[None].to(DEVICE)

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_precision):
            flow_low, flow_pr, info = model(image1, image2, **kwargs)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        rho_list.append(info['sradius'].mean().item())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = np.mean(out_list) * 100

    best['epe'] = min(epe, best['epe'])
    best['f1'] = min(f1, best['f1'])
    print(
        f"Validation KITTI: EPE: {epe:.3f} ({best['epe']:.3f}), F1: {f1:.2f} ({best['f1']:.2f})")

    if np.mean(rho_list) != 0:
        print(f"Spectral radius: {np.mean(rho_list)}")

    return {'kitti-epe': epe, 'kitti-f1': f1}
