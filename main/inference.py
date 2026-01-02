import os
import sys
import os.path as osp
import argparse
import time
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
import cv2
from tqdm import tqdm
import json
from typing import Literal, Union
from PIL import Image
# Fixed by SH Heo (251227) - Replace mmdet with YOLO
from ultralytics import YOLO
from utils.inference_utils import non_max_suppression

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1, dest='num_gpus')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', type=str, default='na')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--demo_dataset', type=str, default='na')
    parser.add_argument('--demo_scene', type=str, default='all')
    parser.add_argument('--show_verts', type=bool, default=False)
    parser.add_argument('--show_bbox', type=bool, default=False)
    parser.add_argument('--save_mesh', type=bool, default=False)
    parser.add_argument('--save_params', type=bool, default=False, help='Save SMPL-X params (.npz) and meta (.json)')
    parser.add_argument('--multi_person', type=bool, default=True)
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--bbox_thr', type=int, default=50)

    parser.add_argument('--pretrained_model', type=str, default='server01_smplerx_small_mscoco_ubody_260102_2/snapshot_2')

    # Fixed by SH Heo (251227)
    parser.add_argument('--video_input', type=str, default=None, help='Input video file path')
    parser.add_argument('--video_output', type=str, default=None, help='Output video file path')
    args = parser.parse_args()
    return args

# Fixed by SH Heo (251227)
def process_video_input(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, total_frames

# Fixed by SH Heo (251227)
def create_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer

# Fixed by SH Heo (251227) - YOLO result processing
def process_yolo_results(results):
    """Convert YOLO results to bbox list format [x1, y1, x2, y2, score]"""
    bboxes = []
    boxes = results[0].boxes
    for i in range(len(boxes)):
        if int(boxes.cls[i]) == 0:  # person class only
            xyxy = boxes.xyxy[i].cpu().numpy()
            score = float(boxes.conf[i].cpu().numpy())
            bboxes.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], score])
    # Sort by area (descending)
    bboxes = sorted(bboxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
    return [bboxes]

def main():

    args = parse_args()
    config_path = osp.join('./config', f'config_{args.pretrained_model}.py')
    ckpt_path = osp.join('../pretrained_models', f'{args.pretrained_model}.pth.tar')

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(args.testset, args.agora_benchmark, shapy_eval_split=None,
                            pretrained_model_path=ckpt_path, use_cache=False)
    cfg.update_config(args.num_gpus, args.exp_name)
    cudnn.benchmark = True

    # load model
    from base import Demoer
    from utils.preprocessing import load_img, process_bbox, generate_patch_image
    from utils.vis import render_mesh, render_mesh_with_texture, save_obj
    import pyrender
    from utils.human_models import smpl_x
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()

    inference_count = 0

    start = int(args.start)
    end = start + int(args.end)
    multi_person = args.multi_person


    # Fixed by SH Heo (251227) - Replace mmdet with YOLO
    detector = YOLO('../pretrained_models/yolo11l.pt')

    # Load texture and UV data for mesh rendering
    texture_path = osp.join('..', 'common', 'texture', 'smplx_texture_m_2023_dressed.png')
    texture_image = Image.open(texture_path)
    uv_template_path = osp.join('..', 'common', 'texture', 'smplx_uv', 'smplx_uv_2023.npz')
    uv_data = np.load(uv_template_path)
    per_vertex_uv = uv_data['uv_coordinates']

    # Create renderer (will be resized per frame if needed)
    mesh_renderer = None

    # Time profiling
    time_yolo = 0
    time_smplerx = 0
    time_render = 0

    # Fixed by SH Heo (251227) - Process single frame
    def process_frame(original_img, frame_idx, transform, save_to_file=True):
        """Process a single frame and return the result image"""
        nonlocal inference_count, mesh_renderer
        nonlocal time_yolo, time_smplerx, time_render

        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        os.makedirs(args.output_folder, exist_ok=True)

        # Fixed by SH Heo (251227) - YOLO inference
        t0 = time.time()
        yolo_results = detector(original_img, verbose=False)
        yolo_box = process_yolo_results(yolo_results)
        time_yolo += time.time() - t0

        if len(yolo_box[0]) < 1:
            return vis_img

        if not multi_person:
            # only select the largest bbox
            num_bbox = 1
            yolo_box_filtered = yolo_box[0]
        else:
            # keep bbox by NMS with iou_thr
            yolo_box_filtered = non_max_suppression(yolo_box[0], args.iou_thr)
            num_bbox = len(yolo_box_filtered)

        ## loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_box_xywh = np.zeros((4))
            yolo_box_xywh[0] = yolo_box_filtered[bbox_id][0]
            yolo_box_xywh[1] = yolo_box_filtered[bbox_id][1]
            yolo_box_xywh[2] = abs(yolo_box_filtered[bbox_id][2] - yolo_box_filtered[bbox_id][0])
            yolo_box_xywh[3] = abs(yolo_box_filtered[bbox_id][3] - yolo_box_filtered[bbox_id][1])

            # skip small bboxes by bbox_thr in pixel
            if yolo_box_xywh[2] < args.bbox_thr or yolo_box_xywh[3] < args.bbox_thr * 3:
                continue

            # for bbox visualization
            start_point = (int(yolo_box_filtered[bbox_id][0]), int(yolo_box_filtered[bbox_id][1]))
            end_point = (int(yolo_box_filtered[bbox_id][2]), int(yolo_box_filtered[bbox_id][3]))

            # Preprocessing
            bbox = process_bbox(yolo_box_xywh, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery (model inference)
            t0 = time.time()
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')
            time_smplerx += time.time() - t0
            inference_count += 1
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            ## save mesh (.obj)
            if args.save_mesh and save_to_file:
                save_path_mesh = os.path.join(args.output_folder, 'mesh')
                os.makedirs(save_path_mesh, exist_ok=True)
                save_obj(mesh, smpl_x.face, os.path.join(save_path_mesh, f'{frame_idx:05}_{bbox_id}.obj'))

            ## save single person param (.npz)
            if save_to_file:
                smplx_pred = {}
                smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3).cpu().numpy()
                smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3).cpu().numpy()
                smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy()
                smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy()
                smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy()
                smplx_pred['leye_pose'] = np.zeros((1, 3))
                smplx_pred['reye_pose'] = np.zeros((1, 3))
                smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10).cpu().numpy()
                smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10).cpu().numpy()
                smplx_pred['transl'] = out['cam_trans'].reshape(-1,3).cpu().numpy()
                save_path_smplx = os.path.join(args.output_folder, 'smplx')
                os.makedirs(save_path_smplx, exist_ok=True)

                npz_path = os.path.join(save_path_smplx, f'{frame_idx:05}_{bbox_id}.npz')
                np.savez(npz_path, **smplx_pred)

            ## render single person mesh with texture
            t0 = time.time()
            focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
            princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]

            # Create or update renderer if needed
            if mesh_renderer is None or mesh_renderer.viewport_width != original_img_width or mesh_renderer.viewport_height != original_img_height:
                if mesh_renderer is not None:
                    mesh_renderer.delete()
                mesh_renderer = pyrender.OffscreenRenderer(viewport_width=original_img_width, viewport_height=original_img_height, point_size=1.0)

            if args.show_verts:
                vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=True)
            else:
                vis_img = render_mesh_with_texture(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt},
                                                   mesh_renderer, texture_image, per_vertex_uv)
            time_render += time.time() - t0

            if args.show_bbox:
                vis_img = cv2.rectangle(vis_img, start_point, end_point, (255, 0, 0), 2)

            ## save single person meta (.json)
            if save_to_file:
                meta = {'focal': focal,
                        'princpt': princpt,
                        'bbox': bbox.tolist(),
                        'bbox_mmdet': yolo_box_xywh.tolist(),
                        'bbox_id': bbox_id,
                        'frame_idx': frame_idx}
                json_object = json.dumps(meta, indent=4)

                save_path_meta = os.path.join(args.output_folder, 'meta')
                os.makedirs(save_path_meta, exist_ok=True)
                with open(os.path.join(save_path_meta, f'{frame_idx:05}_{bbox_id}.json'), "w") as outfile:
                    outfile.write(json_object)

        return vis_img

    # Fixed by SH Heo (251227) - Video mode vs Image mode
    if args.video_input:
        # Video mode: read video directly with OpenCV
        cap, fps, width, height, total_frames = process_video_input(args.video_input)

        # Set output video path
        if args.video_output:
            output_video_path = args.video_output
        else:
            # Default output path: input_output.mp4
            base_name = osp.splitext(args.video_input)[0]
            output_video_path = f"{base_name}_output.mp4"

        # Create output directory if not exists
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Create video writer
        writer = create_video_writer(output_video_path, fps, width, height)

        transform = transforms.ToTensor()

        for frame_idx in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            vis_img = process_frame(original_img, frame_idx, transform, save_to_file=True)

            # Convert RGB to BGR and uint8, then write to video
            vis_img_uint8 = np.clip(vis_img, 0, 255).astype(np.uint8)
            writer.write(vis_img_uint8[:, :, ::-1])

        cap.release()
        writer.release()
        print(f"Output video saved to: {output_video_path}")

        # Print profiling results
        time_total = time_yolo + time_smplerx + time_render
        print(f"\nTotal frames: {total_frames}")
        print(f"Total time: {time_total:.2f}s")
        print(f"Total FPS: {total_frames / time_total:.2f}")
        print(f"  - YOLO: {time_yolo:.2f}s, {total_frames / time_yolo:.2f} FPS")
        print(f"  - SMPLer-X: {time_smplerx:.2f}s, {inference_count / time_smplerx:.2f} FPS")
        print(f"  - Rendering: {time_render:.2f}s, {inference_count / time_render:.2f} FPS")

    else:
        # Image mode: support both single image file and directory
        if os.path.isfile(args.img_path):
            # Single image file
            img_paths = [args.img_path]
        elif os.path.isdir(args.img_path):
            # Directory: read images with numbered format (original behavior)
            img_paths = [os.path.join(args.img_path, f'{int(frame):06d}.jpg') for frame in range(start, end)]
        else:
            raise FileNotFoundError(f"Path does not exist: {args.img_path}")

        for frame, img_path in enumerate(tqdm(img_paths)):
            # prepare input image
            transform = transforms.ToTensor()
            original_img = load_img(img_path)
            vis_img = original_img.copy()
            original_img_height, original_img_width = original_img.shape[:2]
            os.makedirs(args.output_folder, exist_ok=True)

            # Fixed by SH Heo (251227) - YOLO inference
            yolo_results = detector(original_img, verbose=False)
            yolo_box = process_yolo_results(yolo_results)

            # save original image if no bbox
            if len(yolo_box[0])<1:
                frame_name = os.path.basename(img_path)
                os.makedirs(args.output_folder, exist_ok=True)
                cv2.imwrite(os.path.join(args.output_folder, frame_name), vis_img[:, :, ::-1])
                continue

            if not multi_person:
                # only select the largest bbox
                num_bbox = 1
                yolo_box = yolo_box[0]
            else:
                # keep bbox by NMS with iou_thr
                yolo_box = non_max_suppression(yolo_box[0], args.iou_thr)
                num_bbox = len(yolo_box)

            ## loop all detected bboxes
            for bbox_id in range(num_bbox):
                yolo_box_xywh = np.zeros((4))
                yolo_box_xywh[0] = yolo_box[bbox_id][0]
                yolo_box_xywh[1] = yolo_box[bbox_id][1]
                yolo_box_xywh[2] =  abs(yolo_box[bbox_id][2]-yolo_box[bbox_id][0])
                yolo_box_xywh[3] =  abs(yolo_box[bbox_id][3]-yolo_box[bbox_id][1])

                # skip small bboxes by bbox_thr in pixel
                if yolo_box_xywh[2] < args.bbox_thr or yolo_box_xywh[3] < args.bbox_thr * 3:
                    continue

                # for bbox visualization
                start_point = (int(yolo_box[bbox_id][0]), int(yolo_box[bbox_id][1]))
                end_point = (int(yolo_box[bbox_id][2]), int(yolo_box[bbox_id][3]))

                bbox = process_bbox(yolo_box_xywh, original_img_width, original_img_height)
                img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
                img = transform(img.astype(np.float32))/255
                img = img.cuda()[None,:,:,:]
                inputs = {'img': img}
                targets = {}
                meta_info = {}

                # mesh recovery
                with torch.no_grad():
                    out = demoer.model(inputs, targets, meta_info, 'test')
                mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

                ## save mesh
                if args.save_mesh:
                    save_path_mesh = os.path.join(args.output_folder, 'mesh')
                    os.makedirs(save_path_mesh, exist_ok= True)
                    save_obj(mesh, smpl_x.face, os.path.join(save_path_mesh, f'{frame:05}_{bbox_id}.obj'))

                ## save single person param
                if args.save_params:
                    smplx_pred = {}
                    smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['leye_pose'] = np.zeros((1, 3))
                    smplx_pred['reye_pose'] = np.zeros((1, 3))
                    smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10).cpu().numpy()
                    smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10).cpu().numpy()
                    smplx_pred['transl'] =  out['cam_trans'].reshape(-1,3).cpu().numpy()
                    save_path_smplx = os.path.join(args.output_folder, 'smplx')
                    os.makedirs(save_path_smplx, exist_ok= True)

                    npz_path = os.path.join(save_path_smplx, f'{frame:05}_{bbox_id}.npz')
                    np.savez(npz_path, **smplx_pred)

                ## render single person mesh with texture
                focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
                princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]

                # Create or update renderer if needed
                if mesh_renderer is None or mesh_renderer.viewport_width != original_img_width or mesh_renderer.viewport_height != original_img_height:
                    if mesh_renderer is not None:
                        mesh_renderer.delete()
                    mesh_renderer = pyrender.OffscreenRenderer(viewport_width=original_img_width, viewport_height=original_img_height, point_size=1.0)

                if args.show_verts:
                    vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=True)
                else:
                    vis_img = render_mesh_with_texture(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt},
                                                       mesh_renderer, texture_image, per_vertex_uv)
                if args.show_bbox:
                    vis_img = cv2.rectangle(vis_img, start_point, end_point, (255, 0, 0), 2)

                ## save single person meta
                if args.save_params:
                    meta = {'focal': focal,
                            'princpt': princpt,
                            'bbox': bbox.tolist(),
                            'bbox_mmdet': yolo_box_xywh.tolist(),
                            'bbox_id': bbox_id,
                            'img_path': img_path}
                    json_object = json.dumps(meta, indent=4)

                    save_path_meta = os.path.join(args.output_folder, 'meta')
                    os.makedirs(save_path_meta, exist_ok= True)
                    with open(os.path.join(save_path_meta, f'{frame:05}_{bbox_id}.json'), "w") as outfile:
                        outfile.write(json_object)

            ## save rendered image with all person
            frame_name = os.path.basename(img_path)
            os.makedirs(args.output_folder, exist_ok=True)
            cv2.imwrite(os.path.join(args.output_folder, frame_name), vis_img[:, :, ::-1])

    # Cleanup renderer
    if mesh_renderer is not None:
        mesh_renderer.delete()


if __name__ == "__main__":
    main()