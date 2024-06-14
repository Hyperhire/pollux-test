#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, cos_loss, bce_loss, knn_smooth_loss, scale_loss
from gaussian_renderer import render, network_gui
import numpy as np
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, match_depth, normal2curv, resize_image, cross_sample
from torchvision.utils import save_image
from argparse import ArgumentParser, Namespace
import time
import os
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# =========== ljw ===========
from utils.pollux_utils import load_mask_image, load_dilated_mask_image
import yaml
import torch.nn.functional as F
from datetime import datetime as dt
# =========== lsj ===========
use_scale_loss = 1

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    config = args.config
    cfg = yaml.safe_load(open(config))
    pad = cfg['param']['pad']

    first_iter = 0
    tb_writer, unique_str, save_path = prepare_output_and_logger(dataset, args.exp_id)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, opt.camera_lr, shuffle=False, resolution_scales=[1, 2, 4])
    use_mask = dataset.use_mask

    current_datetime = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    cfg['log']['last_train_id'] = unique_str
    cfg['log']['datetime'] = current_datetime
    cfg['log']['saved_path'] = save_path
    with open(config, 'w') as file:
        yaml.dump(cfg, file, default_flow_style=False)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif use_mask: # visual hull init
        gaussians.mask_prune(scene.getTrainCameras(), 4)
        None

    opt.densification_interval = max(opt.densification_interval, len(scene.getTrainCameras()))

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    pool = torch.nn.MaxPool2d(9, stride=1, padding=4)

    #=============== Prepare dilated mask dictionary per scale (ljw, lsj) ==================
   
    dilated_mask_per_scale = {
        '1': {},
        '2': {},
        '4': {},
    }
    for scale in dilated_mask_per_scale.keys():
        print(f"dilated_mask processing start at scale = {scale}")
        for iter in range(len(scene.getTrainCameras())):
            scale_num = int(scale)
            viewpoint_cam_temp = scene.getTrainCameras(scale_num)[iter]
            dilated_mask = load_dilated_mask_image(mask_path=args.mask_path, iteration=viewpoint_cam_temp.image_name, resolution=scale_num)
            dilated_mask_per_scale[f'{scale}'][viewpoint_cam_temp.image_name]=dilated_mask
            # visualization
            # Squeeze the mask if it has a leading dimension of size 1
            #if dilated_mask.ndim == 3 and dilated_mask.shape[0] == 1:
            #    dilated_mask = dilated_mask.squeeze(0)
            #plt.imshow(dilated_mask)
            #plt.savefig(f"./test/mask_{viewpoint_cam_temp.image_name}.png")
            #plt.close()
    #=========================================================================================

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    count = -1
    for iteration in range(first_iter, opt.iterations + 2):

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if iteration - 1 == 0:
            scale = 4
        # scale = 1

        # Pick a random Camera
        if not viewpoint_stack:
            if iteration - 1 == 0:
                scale = 4
            elif iteration - 1 == 2000:
                scale = 2
            elif iteration - 1 >= 5000:
                scale = 1
            viewpoint_stack = scene.getTrainCameras(scale).copy()[:]
            data_len = len(viewpoint_stack)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        # viewpoint_cam = scene.getTrainCameras(scale)[0]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        background = torch.rand((3), dtype=torch.float32, device="cuda") if dataset.random_background else background
        patch_size = [float('inf'), float('inf')]

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, patch_size)
        image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        mask_gt = viewpoint_cam.get_gtMask(use_mask) #Original code

        ##### mask_gt code ####
        #mask_gt = dilated_mask_per_scale[str(scale)][viewpoint_cam.image_name]
        ## Transfer to the GPU 
        #mask_gt = mask_gt.float().cuda()
        ## Apply pooling operation
        #pool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)  # Example parameters
        #pooled_mask = pool(mask_gt)

        gt_image = viewpoint_cam.get_gtImage(background, use_mask) #Original code
        mask_vis = (opac.detach() > 1e-5)
        normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
        d2n = depth2normal(depth, mask_vis, viewpoint_cam)
        mono = viewpoint_cam.mono if dataset.mono_normal else None
        if mono is not None:
            #### mask_gt code ####
            #mono = mono.cuda() 
            #mono *= pooled_mask
            mono *= mask_gt # Original code
            monoN = mono[:3]
            # monoD = mono[3:]
            # monoD_match, mask_match = match_depth(monoD, depth, mask_gt * mask_vis, 256, [viewpoint_cam.image_height, viewpoint_cam.image_width])

        #=================== mask_gt and gt_image visualization (ljw, lsj)  ===============

        #print(f"viewpoint_cam.image_name : {viewpoint_cam.image_name}")
        #print(f"mask_gt : {mask_gt}")
        #print(f"mask_gt shape : {mask_gt.shape}")
        #print(f"gt_image : {gt_image}")
        #print(f"gt_image shape : {gt_image.shape}")
        ## Transfer tensors to CPU
        #mask_gt_np = mask_gt.cpu().numpy()
        #mask_gt_np = mask_gt_np.squeeze()
        #gt_image_np = gt_image.cpu().numpy()
        #gt_image_np = np.transpose(gt_image_np, (1, 2, 0))
        #plt.imshow(mask_gt_np, cmap='gray')
        #plt.savefig(f"./test/masked_gt_{viewpoint_cam.image_name}.png")
        #plt.close()    
        #plt.imshow(gt_image_np)
        #plt.savefig(f"./test/gt_image{viewpoint_cam.image_name}.png")
        #plt.close()    
        ##==================================================================================

        
        # =========== visible gaussian points ===========
        visible, scrPos = gaussians.seg_mask_prune([viewpoint_cam], pad) # default pad : 4

        # print(f"scrPos : {scrPos.shape}")
        
        indices = torch.nonzero(visible, as_tuple=True)[0]

        visible_mask = torch.zeros(scrPos.shape[1], dtype=torch.bool)
        visible_mask[indices] = True

        filtered_scrPos = torch.full_like(scrPos, -1)
        filtered_scrPos[:, visible_mask, :] = scrPos[:, visible_mask, :]

        # print(f"filtered_scrPos : {filtered_scrPos.shape}")

        scrPos_np = filtered_scrPos.cpu().numpy()[0]
        image_np = gt_image.cpu().numpy().transpose(1, 2, 0)

        # if iteration % 100 == 0:
        #     plt.imshow(image_np)
        #     plt.scatter(scrPos_np[:, 0], scrPos_np[:, 1], c='red', s=0.001)
        #     plt.savefig("./test/projected.png")
            
        # =========== SAM Mask (ljw, lsj) ===========
        #if iteration > 29000:
        #    SAM_mask = load_dilated_mask_image(args.mask_path, iteration=viewpoint_cam.image_name, resolution=scale)
        #else:
        #    SAM_mask = load_mask_image(args.mask_path, iteration=viewpoint_cam.image_name, resolution=scale)
        SAM_mask = dilated_mask_per_scale[str(scale)][viewpoint_cam.image_name] 
        SAM_mask = SAM_mask.cuda()

        image = image * SAM_mask
        gt_image = gt_image * SAM_mask
        mask_gt = mask_gt * SAM_mask 
        normal = normal * SAM_mask
        monoN = monoN * SAM_mask
        d2n = d2n * SAM_mask
        opac = opac * SAM_mask

        # =========== Visible points & SAM Mask (ljw, lsj) ===================
        scrPos_np_int = scrPos_np.astype(int)

        # Create a Boolean tensor of the same shape as scrPos_np with default value True
        mask_valid = torch.ones(scrPos_np.shape[0], dtype=torch.bool).cuda()

        # Check the SAM_mask for each point in scrPos_np
        # Using vectorized operations for performance
        x_coords = scrPos_np_int[:, 0]
        y_coords = scrPos_np_int[:, 1]

        valid_x = (x_coords >= 0) & (x_coords < SAM_mask.shape[2])
        valid_y = (y_coords >= 0) & (y_coords < SAM_mask.shape[1])

        valid_coords = valid_x & valid_y

        x_coords = x_coords[valid_coords]
        y_coords = y_coords[valid_coords]
        indices = torch.arange(scrPos_np.shape[0], device='cuda')[valid_coords]

        mask_valid[indices] = SAM_mask[0, y_coords, x_coords] != 0

        # if iteration % 100 == 0:
        #     plt.clf()
        #     plt.imshow(image_np)
        #     plt.scatter(scrPos_np[mask_valid.cpu(), 0], scrPos_np[mask_valid.cpu(), 1], c='green', s=10, label='Inside SAM_mask')
        #     plt.scatter(scrPos_np[~mask_valid.cpu(), 0], scrPos_np[~mask_valid.cpu(), 1], c='red', s=10, label='Outside SAM_mask')
        #     plt.title("SAM Mask & Visible Points")
        #     # plt.legend()
        #     plt.savefig("./test/projected_with_sam_mask.png")

        # =========== Loss ===========
        Ll1 = l1_loss(image, gt_image)
        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss_mask = (opac * (1 - pool(mask_gt))).mean()
        
        if mono is not None:
            loss_monoN = cos_loss(normal, monoN, weight=mask_gt)
            # loss_depth = l1_loss(depth * mask_match, monoD_match)

        loss_surface = cos_loss(resize_image(normal, 1), resize_image(d2n, 1), thrsh=np.pi*1/10000 , weight=1)
        
        opac_ = gaussians.get_opacity - 0.5
        opac_mask = torch.gt(opac_, 0.01) * torch.le(opac_, 0.99)
        loss_opac = torch.exp(-(opac_ * opac_) * 20)
        loss_opac = (loss_opac * opac_mask).mean()
        
        curv_n = normal2curv(normal, mask_vis)
        # curv_d2n = normal2curv(d2n, mask_vis_2)
        curv_n = curv_n * SAM_mask
        # curv_d2n = normal2curv(d2n, mask_vis_2)
        loss_curv = l1_loss(curv_n * 1, 0) #+ 1 * l1_loss(curv_d2n, 0)
        
        loss = 1 * loss_rgb
        loss += 1 * loss_mask
        loss += (0.01 + 0.1 * min(2 * iteration / opt.iterations, 1)) * loss_surface
        # loss += (0.00 + 0.1 * min(1 * iteration / opt.iterations, 1)) * loss_surface
        loss += (0.005 - ((iteration / opt.iterations)) * 0.0) * loss_curv
        loss += loss_opac * 0.01


        #================= Scale loss for Gaussian scale minimize (ljw, lsj) ==============
        if use_scale_loss:
            loss += scale_loss(scales = gaussians.get_scaling, lambda_flatten = 100.0) 
        #================================================================================


        # mono = None
        if mono is not None:
            loss += (0.04 - ((iteration / opt.iterations)) * 0.03) * loss_monoN
            # loss += 0.01 * loss_depth

        loss.backward()

        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_rgb.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, Pts={len(gaussians._xyz)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            test_background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, pipe, test_background, use_mask)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration > opt.densify_from_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                min_opac = 0.1# if iteration <= opt.densify_from_iter else 0.1
                # min_opac = 0.05 if iteration <= opt.densify_from_iter else 0.005

                # Binary_mask로 background random gaussian 삭제 
                if iteration % opt.pruning_interval == 0 and iteration > 29700: # TODO : Visualization 함수 만들기
                    #gaussians.adaptive_prune(min_opac, scene.cameras_extent) # Original code
                    
                    plt.clf()
                    plt.imshow(image_np)
                    plt.scatter(scrPos_np[mask_valid.cpu(), 0], scrPos_np[mask_valid.cpu(), 1], c='green', s=10, label='Inside SAM_mask')
                    plt.scatter(scrPos_np[~mask_valid.cpu(), 0], scrPos_np[~mask_valid.cpu(), 1], c='red', s=10, label='Outside SAM_mask')
                    plt.title("SAM Mask & Visible Points")
                    # plt.legend()
                    plt.savefig(f"./test/{iteration}_before_projected_with_sam_mask.png")

                    gaussians.adaptive_prune_modified(min_opac, scene.cameras_extent, mask_valid) # Modified
                    gaussians.adaptive_densify_without_clone_and_split(opt.densify_grad_threshold, scene.cameras_extent) # Modified

                    # =======================================
                    visible, scrPos = gaussians.seg_mask_prune([viewpoint_cam], pad) # default pad : 4

                    # print(f"scrPos : {scrPos.shape}")
                    
                    indices = torch.nonzero(visible, as_tuple=True)[0]

                    visible_mask = torch.zeros(scrPos.shape[1], dtype=torch.bool)
                    visible_mask[indices] = True

                    filtered_scrPos = torch.full_like(scrPos, -1)
                    filtered_scrPos[:, visible_mask, :] = scrPos[:, visible_mask, :]

                    # print(f"filtered_scrPos : {filtered_scrPos.shape}")

                    scrPos_np = filtered_scrPos.cpu().numpy()[0]
                    image_np = gt_image.cpu().numpy().transpose(1, 2, 0)

                    # if iteration % 100 == 0:
                    #     plt.imshow(image_np)
                    #     plt.scatter(scrPos_np[:, 0], scrPos_np[:, 1], c='red', s=0.001)
                    #     plt.savefig("./test/projected.png")
                        
                    # =========== SAM Mask ===========

                    image = image * SAM_mask
                    gt_image = gt_image * SAM_mask
                    mask_gt = mask_gt * SAM_mask
                    normal = normal * SAM_mask
                    monoN = monoN * SAM_mask
                    d2n = d2n * SAM_mask
                    opac = opac * SAM_mask

                    # =========== Visible points & SAM Mask ===========
                    scrPos_np_int = scrPos_np.astype(int)

                    # Create a Boolean tensor of the same shape as scrPos_np with default value True
                    mask_valid = torch.ones(scrPos_np.shape[0], dtype=torch.bool).cuda()

                    # Check the SAM_mask for each point in scrPos_np
                    # Using vectorized operations for performance
                    x_coords = scrPos_np_int[:, 0]
                    y_coords = scrPos_np_int[:, 1]

                    valid_x = (x_coords >= 0) & (x_coords < SAM_mask.shape[2])
                    valid_y = (y_coords >= 0) & (y_coords < SAM_mask.shape[1])

                    valid_coords = valid_x & valid_y

                    x_coords = x_coords[valid_coords]
                    y_coords = y_coords[valid_coords]
                    indices = torch.arange(scrPos_np.shape[0], device='cuda')[valid_coords]

                    mask_valid[indices] = SAM_mask[0, y_coords, x_coords] != 0
                    # ================================================================================

                    plt.clf()
                    plt.imshow(image_np)
                    plt.scatter(scrPos_np[mask_valid.cpu(), 0], scrPos_np[mask_valid.cpu(), 1], c='green', s=10, label='Inside SAM_mask')
                    plt.scatter(scrPos_np[~mask_valid.cpu(), 0], scrPos_np[~mask_valid.cpu(), 1], c='red', s=10, label='Outside SAM_mask')
                    plt.title("SAM Mask & Visible Points")
                    # plt.legend()
                    plt.savefig(f"./test/{iteration}_after_projected_with_sam_mask.png")

                if iteration % opt.densification_interval == 0 and iteration < 29700:
                    gaussians.adaptive_prune(min_opac, scene.cameras_extent) # Original code
                    gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent) # Original code

                # TODO : reset opacity interval 왜하는지 알아내기
                if (iteration - 1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0 and iteration < 29700:
                    gaussians.reset_opacity(0.12, iteration)



            if (iteration - 1) % 1000 == 0:

                if mono is not None:
                    monoN_wrt = normal2rgb(monoN, mask_gt)
                    # monoD_wrt = depth2rgb(monoD_match, mask_match)

                normal_wrt = normal2rgb(normal, mask_vis)
                depth_wrt = depth2rgb(depth, mask_vis)
                img_wrt = torch.cat([gt_image, image, normal_wrt * opac, depth_wrt * opac], 2)
                save_image(img_wrt.cpu(), f'test/test.png')

            
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad()
                # viewpoint_cam.optimizer.step()
                # viewpoint_cam.optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                # gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                # scene.visualize_cameras()
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args, exp_id):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())

        args.model_path = os.path.join("./output/test608", f"{args.source_path.split('/')[-1]}_{unique_str[0:10]+exp_id}")
        
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, unique_str[0:10]+exp_id, args.model_path

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, pipe, bg, use_mask):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()[::8]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render(viewpoint, scene.gaussians, pipe, bg, [float('inf'), float('inf')])["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.get_gtImage(bg, with_mask=use_mask), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # ljw
    parser.add_argument('--config', type=str, default='./config/train.yaml', help='Path to the configuration file')
    parser.add_argument("--exp_id", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    
    # ljw -> TODO : config(.yaml) file setting
    args.data_path = os.path.join("./data")
    args.mask_path = os.path.join(args.data_path, 'binary_mask_npy/binary_mask_npy')

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
