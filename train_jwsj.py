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
from utils.image_utils import world2scrn_two
import ChamferDistancePytorch.chamfer2D.dist_chamfer_2D as chamfer

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    # 해당 Config파일은 ./config/train.yaml에 있습니다.
    # 이 yaml파일에서 필요한 파라미터들을 관리할 수 있습니다.
    config = args.config
    cfg = yaml.safe_load(open(config))
    pad = cfg['param']['pad']
    prune_after_iteration = cfg['param']['prune_after_iteration']
    # =========== lsj ===========
    use_scale_loss = cfg['train']['use_scale_loss']
    use_SAM_mask = cfg['train']['use_SAM_mask']
    use_dist_loss = cfg['train']['use_dist_loss']


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
            dilated_mask = load_mask_image(mask_path=args.mask_path, iteration=viewpoint_cam_temp.image_name, resolution=scale_num)
            dilated_mask_per_scale[f'{scale}'][viewpoint_cam_temp.image_name]=dilated_mask
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

        gt_image = viewpoint_cam.get_gtImage(background, use_mask) #Original code
        mask_vis = (opac.detach() > 1e-5)
        normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
        d2n = depth2normal(depth, mask_vis, viewpoint_cam)
        mono = viewpoint_cam.mono if dataset.mono_normal else None
        if mono is not None:
            #### mask_gt code ####
            mono *= mask_gt # Original code
            monoN = mono[:3]

        # =========== visible gaussian points ===========
            # 목표 :
            # 이미지 내부에 해당하는 gs points에서
            # SAM 밖의 포인트와 SAM내부 포인트 간의 거리를 dist loss로 부여
        if use_SAM_mask:
            vis = True

            # visible: 이미지 내에 해당하는지 여부인 mask
            # scrPos: 3D GS points가 2D에 투영된 좌표(이미지 크기를 넘는 포인트들이 있을 수 있다)
            # 그러면 상식적으로 visible.shape == scrPos.shape
            # 확인 : visible, scrPos : (torch.Size([15848]), torch.Size([1(채널), 15848(개수), 2(y, x)]))
            # visible.sum() : 15812 (True cases)
            visible, scrPos = gaussians.seg_mask_prune([viewpoint_cam], pad) # default pad : 4

            # visible에 해당하는 screen position만 남기기
            # CUDA : visible_scrPos : torch.Size([15812, 2])
            visible_scrPos = scrPos[0][visible]

            # SAM_mask : torch.Size([3, 320, 180])
            SAM_mask = dilated_mask_per_scale[str(scale)][viewpoint_cam.image_name] 
            SAM_mask = SAM_mask.cuda()

            # ~ = All_y, All_x
            visible_pos_to_yx_coord = visible_scrPos[:, 1], visible_scrPos[:, 0] 
            # SAM_mask_with_visible : torch.Size([15812])
            SAM_mask_with_visible = SAM_mask[0][visible_pos_to_yx_coord]

            # 15812[15812]
            inside_sam = visible_scrPos[SAM_mask_with_visible]
            outside_sam = visible_scrPos[~SAM_mask_with_visible]

            # cdist : dist벡터화 계산 >>> backward 안먹힘 * cdist의 유명한 문제
            #dist_matrix = torch.cdist(outside_sam.unsqueeze(0).float(), inside_sam.unsqueeze(0).float(), p=2).squeeze(0)

            # chamferDist
            chamLoss = chamfer.chamfer_2DDist()
            _, __, idx1, idx2 = chamLoss(inside_sam, outside_sam)
    
            # 각 outside_point에 대해 가장 가까운 inside_point와의 거리 선택
            min_dist, min_indices = torch.min(idx1, dim=1)
            
            # 거리 손실
            distance_loss = min_dist.sum()
            
            print(f'Distance Loss: {distance_loss.item()}')

            # print(f"inside_sam : {inside_sam}")
            # print(f"inside_sam : {inside_sam.shape}") # 12875
            # print(f"outside_sam : {outside_sam}")
            # print(f"outside_sam : {outside_sam.shape}") # 2937


            if vis:
                # GT-Image : CUDA to cpu
                gt_image_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                # CUDA to cpu (for vis)
                visible_scrPos_np = visible_scrPos.detach().cpu().numpy()

                fig, axs = plt.subplots(1, 2, figsize=(20, 10))
                axs[0].imshow(gt_image_np)
                axs[0].scatter(visible_scrPos_np[:, 0], visible_scrPos_np[:, 1], c='green', s=10)  # 초록색 점으로 표시
                axs[0].set_title('Gaussian Points on GT Image')
                axs[0].set_xlabel('X-axis')
                axs[0].set_ylabel('Y-axis')

                axs[1].imshow(gt_image_np)
                axs[1].scatter(inside_sam[:, 0].cpu(), inside_sam[:, 1].cpu(), c='green', s=10, label='Inside SAM')
                axs[1].scatter(outside_sam[:, 0].cpu(), outside_sam[:, 1].cpu(), c='red', s=10, label='Outside SAM')

                # outside_sam 포인트와 가장 가까운 inside_sam 포인트 간의 선 그리기
                for i, index in enumerate(min_indices):
                    outside_point = outside_sam[i].cpu()
                    inside_point = inside_sam[index].cpu()
                    axs[1].plot([outside_point[0], inside_point[0]], [outside_point[1], inside_point[1]], 'b-', linewidth=0.5)

                axs[1].set_title('Gaussian Points on GT Image')
                axs[1].set_xlabel('X-axis')
                axs[1].set_ylabel('Y-axis')
                axs[1].legend()

                plt.savefig("./test/my_combined_test.png")
                plt.show()



            

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

        # if use_SAM_mask:
        #     curv_n = curv_n * SAM_mask
        
        loss_curv = l1_loss(curv_n * 1, 0) #+ 1 * l1_loss(curv_d2n, 0)
        
        loss = 1 * loss_rgb
        loss += 1 * loss_mask
        loss += (0.01 + 0.1 * min(2 * iteration / opt.iterations, 1)) * loss_surface
        # loss += (0.00 + 0.1 * min(1 * iteration / opt.iterations, 1)) * loss_surface
        loss += (0.005 - ((iteration / opt.iterations)) * 0.0) * loss_curv
        loss += loss_opac * 0.01


        #================= Scale loss for Gaussian scale minimize (ljw, lsj) ========================================
        if use_scale_loss:
            loss += scale_loss(scales = gaussians.get_scaling, lambda_flatten = 100.0) 
        #============================================================================================================

        if use_dist_loss:
            loss += distance_loss

        # mono = None
        if mono is not None:
            loss += (0.04 - ((iteration / opt.iterations)) * 0.03) * loss_monoN
            # loss += 0.01 * loss_depth

        loss.backward()
        # print("Gradients for world_xyz:", world_xyz.grad)  # Check gradients for world_xyz
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_rgb.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, Pts={len(gaussians._xyz)}, Total={loss:.{7}f}"})
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
            # if iteration > opt.densify_from_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            #     min_opac = 0.1# if iteration <= opt.densify_from_iter else 0.1
            #     # min_opac = 0.05 if iteration <= opt.densify_from_iter else 0.005

            #     # Binary_mask로 background random gaussian 삭제 
            #     # if iteration % opt.pruning_interval == 0
                
            #     if iteration % opt.pruning_interval == 0 and iteration > prune_after_iteration : # TODO : Visualization 함수 만들기
            #         #gaussians.adaptive_prune(min_opac, scene.cameras_extent) # Original code
            #         gaussians.adaptive_prune_modified(min_opac, scene.cameras_extent, mask_valid) # Modified
            #         gaussians.adaptive_densify_without_clone_and_split(opt.densify_grad_threshold, scene.cameras_extent) # Modified

            #     if iteration % opt.densification_interval == 0 and iteration < (opt.iterations-300):
            #         gaussians.adaptive_prune(min_opac, scene.cameras_extent) # Original code
            #         gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent) # Original code

            #     # TODO : reset opacity interval 왜하는지 알아내기
            #     if (iteration - 1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0 and iteration < (opt.iterations-300):
            #         gaussians.reset_opacity(0.12, iteration)

            # Densification
            if iteration > opt.densify_from_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                min_opac = 0.1# if iteration <= opt.densify_from_iter else 0.1
                # min_opac = 0.05 if iteration <= opt.densify_from_iter else 0.005
                # if iteration % opt.pruning_interval == 0:
                if iteration % opt.densification_interval == 0:
                    gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                    gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent)
                
                if (iteration - 1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0:
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

        args.model_path = os.path.join("./output/test615", f"{args.source_path.split('/')[-1]}_{unique_str[0:10]+exp_id}")
        
        
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
