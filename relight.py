import os
import sys
from argparse import ArgumentParser
from os import makedirs
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from hugs.utils.general import safe_state
from hugs.utils.image import viridis_cmap
import nvdiffrast.torch as dr

from hugs.datasets import NeumanDataset
from hugs.models.hugs_wo_trimlp import HUGS_WO_TRIMLP
from omegaconf import OmegaConf

import argparse
from loguru import logger
from omegaconf import OmegaConf

# sys.path.append('.')

from hugs.utils.config import get_cfg_items
from hugs.cfg.config import cfg as default_cfg
from hugs.utils.general import safe_state, find_cfg_diff
from hugs.renderer.gs_renderer import render_human_scene

import math

def get_canonical_rays(data) -> torch.Tensor:
        # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
        # TODO: inject intrinsic
        H, W = data['image_height'], data['image_width']
        cen_x = W / 2
        cen_y = H / 2
        tan_fovx = math.tan(data['fovx'] * 0.5)
        tan_fovy = math.tan(data['fovy'] * 0.5)
        focal_x = W / (2.0 * tan_fovx)
        focal_y = H / (2.0 * tan_fovy)

        x, y = torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            indexing="xy",
        )
        x = x.flatten()  # [H * W]
        y = y.flatten()  # [H * W]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cen_x + 0.5) / focal_x,
                    (y - cen_y + 0.5) / focal_y,
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )  # [H * W, 3]
        # NOTE: it is not normalized
        return camera_dirs.cuda()


def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda")
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode="linear")[0]
    return cubemap


def render_set(
    cfg,
    dataset,
    human_gs,
    model_path: str,
    name: str,
    light_name: str,
    hdri: torch.Tensor,
    light: CubemapLight,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = True,
) -> None:
    # build mip for environment light
    light.build_mips()
    envmap = light.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    os.makedirs(os.path.join(model_path, name), exist_ok=True)
    envmap_path = os.path.join(model_path, name, "envmap_relight.png")
    torchvision.utils.save_image(envmap, envmap_path)

    relight_path = os.path.join(model_path, name, "relight")
    makedirs(relight_path, exist_ok=True)

    brdf_lut = get_brdf_lut().cuda()

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    for idx, data in enumerate(tqdm(dataset, desc="Validation")):
        bg_color[...] = 0.0  # NOTE: set zero
        
        canonical_rays = get_canonical_rays(data)
        
        human_gs_out, scene_gs_out = None, None
        render_mode = cfg.mode
            
        human_gs_out = human_gs.forward(
            global_orient=data['global_orient'], 
            body_pose=data['body_pose'], 
            betas=data['betas'], 
            transl=data['transl'], 
            smpl_scale=data['smpl_scale'][None],
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
        )
                    
        render_pkg = render_human_scene(
            data=data, 
            human_gs_out=human_gs_out, 
            scene_gs_out=scene_gs_out, 
            bg_color=bg_color,
            render_mode=render_mode,
        )
            
        depth_map = render_pkg["depth_map"]

        depth_img = viridis_cmap(depth_map.squeeze().cpu().numpy())
        depth_img = (depth_img * 255).astype(np.uint8)
        normal_map = render_pkg["normal_map"]
        normal_mask = render_pkg["normal_mask"]

        # normal from point cloud
        H, W = data['image_height'], data['image_width']
        c2w = torch.inverse(data['world_view_transform'].T)  # [4, 4]
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]
        alpha_mask = data['mask'].unsqueeze(0).cuda()

        albedo_map = render_pkg["albedo_map"]  # [3, H, W]
        roughness_map = render_pkg["roughness_map"]  # [1, H, W]
        metallic_map = render_pkg["metallic_map"]  # [1, H, W]
        pbr_result = pbr_shading(
            light=light,
            normals=normal_map.permute(1, 2, 0),  # [H, W, 3]
            view_dirs=view_dirs,
            mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
            albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
            roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
            metallic=metallic_map.permute(1, 2, 0) if metallic else None,  # [H, W, 1]
            tone=tone,
            gamma=gamma,
            brdf_lut=brdf_lut,
        )
        render_rgb = pbr_result["render_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)  # [3, H, W]

        render_rgb = render_rgb * alpha_mask

        torchvision.utils.save_image(render_rgb, os.path.join(relight_path, f"{idx:05d}_{light_name}.png"))


@torch.no_grad()
def launch(
    cfg,
    model_path: str,
    skip_train: bool = False,
    skip_test: bool = False,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = True,
) -> None:
    dataset = NeumanDataset(cfg.dataset.seq, 'val', cfg.mode)
    human_gs = HUGS_WO_TRIMLP(3)

    init_betas = torch.stack([x['betas'] for x in dataset.cached_data], dim=0)
    human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
    human_gs.initialize()

    # load hdri
    hdri_path = cfg.hdri
    print(f"read hdri from {hdri_path}")
    hdri = read_hdr(hdri_path)
    hdri = torch.from_numpy(hdri).cuda()
    res = 256
    cubemap = CubemapLight(base_res=res).cuda()
    cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
    cubemap.eval()

    light_name = os.path.basename(hdri_path).split(".")[0]

    checkpoint = torch.load(cfg.human.ckpt)
    if isinstance(checkpoint, Tuple):
        model_params = checkpoint[0]
    elif isinstance(checkpoint, Dict):
        model_params = checkpoint["gaussians"]
    else:
        raise TypeError
    human_gs.restore(model_params)

    if not skip_train:
        render_set(
            cfg=cfg,
            dataset=dataset,
            human_gs=human_gs,
            model_path=model_path,
            name="train",
            light_name=light_name,
            hdri=hdri,
            light=cubemap,
            metallic=metallic,
            tone=tone,
            gamma=gamma,
        )
    if not skip_test:
        render_set(
            cfg=cfg,
            dataset=dataset,
            human_gs=human_gs,
            model_path=model_path,
            name="test",
            light_name=light_name,
            hdri=hdri,
            light=cubemap,
            metallic=metallic,
            tone=tone,
            gamma=gamma,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True, help="path to the yaml config file")
    parser.add_argument("--cfg_id", type=int, default=-1, help="id of the config to run")
    args, extras = parser.parse_known_args()
    
    cfg_file = OmegaConf.load(args.cfg_file)
    list_of_cfgs, hyperparam_search_keys = get_cfg_items(cfg_file)
    
    logger.info(f'Running {len(list_of_cfgs)} experiments')

    # Initialize system state (RNG)
    safe_state(False)
    
    if args.cfg_id >= 0:
        cfg_item = list_of_cfgs[args.cfg_id]
        logger.info(f'Running experiment {args.cfg_id} -- {cfg_item.exp_name}')
        default_cfg.cfg_file = args.cfg_file
        cfg = OmegaConf.merge(default_cfg, cfg_item, OmegaConf.from_cli(extras))
        launch(cfg=cfg, model_path='/home/jamesyoung0623/ml-hugs/output/human/neuman/lab/hugs_wo_trimlp/test')
    else:
        for exp_id, cfg_item in enumerate(list_of_cfgs):
            logger.info(f'Running experiment {exp_id} -- {cfg_item.exp_name}')
            default_cfg.cfg_file = args.cfg_file
            cfg = OmegaConf.merge(default_cfg, cfg_item, OmegaConf.from_cli(extras))
            launch(cfg=cfg, model_path='/home/jamesyoung0623/ml-hugs/output/human/neuman/lab/hugs_wo_trimlp/test')
  





