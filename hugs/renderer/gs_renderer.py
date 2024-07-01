# Code adapted from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from hugs.utils.spherical_harmonics import SH2RGB
from hugs.utils.rotations import quaternion_to_matrix


def render_human_scene(
    data, 
    human_gs_out,
    scene_gs_out,
    bg_color, 
    human_bg_color=None,
    scaling_modifier=1.0, 
    render_mode='human_scene',
    render_human_separate=False,
):

    feats = None
    if render_mode == 'human_scene':
        feats = torch.cat([human_gs_out['shs'], scene_gs_out['shs']], dim=0)
        means3D = torch.cat([human_gs_out['xyz'], scene_gs_out['xyz']], dim=0)
        opacity = torch.cat([human_gs_out['opacity'], scene_gs_out['opacity']], dim=0)
        scales = torch.cat([human_gs_out['scales'], scene_gs_out['scales']], dim=0)
        rotations = torch.cat([human_gs_out['rotq'], scene_gs_out['rotq']], dim=0)
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'human':
        feats = human_gs_out['shs']
        means3D = human_gs_out['xyz']
        opacity = human_gs_out['opacity']
        albedo = human_gs_out['albedo']
        roughness = human_gs_out['roughness']
        metallic = human_gs_out['metallic']
        scales = human_gs_out['scales']
        rotations = human_gs_out['rotq']
        normals = human_gs_out['normals']
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'scene':
        feats = scene_gs_out['shs']
        means3D = scene_gs_out['xyz']
        opacity = scene_gs_out['opacity']
        scales = scene_gs_out['scales']
        rotations = scene_gs_out['rotq']
        active_sh_degree = scene_gs_out['active_sh_degree']
    else:
        raise ValueError(f'Unknown render mode: {render_mode}')
    
    render_pkg = render(
        means3D=means3D,
        feats=feats,
        opacity=opacity,
        albedo = albedo,
        roughness = roughness,
        metallic = metallic,
        scales=scales,
        rotations=rotations,
        normals=normals,
        data=data,
        scaling_modifier=scaling_modifier,
        bg_color=bg_color,
        active_sh_degree=active_sh_degree,
    )
        
    if render_human_separate and render_mode == 'human_scene':
        render_human_pkg = render(
            means3D=human_gs_out['xyz'],
            feats=human_gs_out['shs'],
            opacity=human_gs_out['opacity'],
            scales=human_gs_out['scales'],
            rotations=human_gs_out['rotq'],
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=human_bg_color if human_bg_color is not None else bg_color,
            active_sh_degree=human_gs_out['active_sh_degree'],
            derive_normal=True,
        )
        render_pkg['human_img'] = render_human_pkg['render']
        render_pkg['human_visibility_filter'] = render_human_pkg['visibility_filter']
        render_pkg['human_radii'] = render_human_pkg['radii']
        
    if render_mode == 'human':
        render_pkg['human_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['human_radii'] = render_pkg['radii']
    elif render_mode == 'human_scene':
        human_n_gs = human_gs_out['xyz'].shape[0]
        scene_n_gs = scene_gs_out['xyz'].shape[0]
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter'][human_n_gs:]
        render_pkg['scene_radii'] = render_pkg['radii'][human_n_gs:]
        if not 'human_visibility_filter' in render_pkg.keys():
            render_pkg['human_visibility_filter'] = render_pkg['visibility_filter'][:-scene_n_gs]
            render_pkg['human_radii'] = render_pkg['radii'][:-scene_n_gs]
            
    elif render_mode == 'scene':
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['scene_radii'] = render_pkg['radii']
        
    return render_pkg
    
    
def render(means3D, feats, opacity, albedo, roughness, metallic, scales, rotations, normals, data, scaling_modifier=1.0, bg_color=None, active_sh_degree=0, override_color=None, inference=False, pad_normal=False, derive_normal=False):
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
        
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        pass

    means2D = screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)

    shs, rgb = None, None
    if len(feats.shape) == 2:
        rgb = feats
    else:
        shs = feats
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=active_sh_degree,
        campos=data['camera_center'],
        prefiltered=False,
        debug=False,
        inference=inference,
        argmax_depth=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    assert albedo.shape[0] == roughness.shape[0] and albedo.shape[0] == metallic.shape[0]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    if override_color is not None:
        colors_precomp = override_color
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    (
        rendered_image,
        radii,
        opacity_map,
        depth_map,
        normal_map_from_depth,
        normal_map,
        albedo_map,
        roughness_map,
        metallic_map,
    ) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        normal=normals,
        shs=shs,
        colors_precomp=colors_precomp,
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        derive_normal=derive_normal,
    )

    normal_mask = (normal_map != 0).all(0, keepdim=True)
    normal_from_depth_mask = (normal_map_from_depth != 0).all(0)
    if pad_normal:
        opacity_map = torch.where(  # NOTE: a trick to filter out 1 / 255
            opacity_map < 0.004,
            torch.zeros_like(opacity_map),
            opacity_map,
        )
        opacity_map = torch.where(  # NOTE: a trick to filter out 1 / 255
            opacity_map > 1.0 - 0.004,
            torch.ones_like(opacity_map),
            opacity_map,
        )
        normal_bg = torch.tensor([0.0, 0.0, 1.0], device=normal_map.device)
        normal_map = normal_map * opacity_map + (1.0 - opacity_map) * normal_bg[:, None, None]
        mask_from_depth = (normal_map_from_depth == 0.0).all(0, keepdim=True).float()
        normal_map_from_depth = normal_map_from_depth * (1.0 - mask_from_depth) + mask_from_depth * normal_bg[:, None, None]

    normal_map_from_depth = torch.where(
        torch.norm(normal_map_from_depth, dim=0, keepdim=True) > 0,
        F.normalize(normal_map_from_depth, dim=0, p=2),
        normal_map_from_depth,
    )
    normal_map = torch.where(
        torch.norm(normal_map, dim=0, keepdim=True) > 0,
        F.normalize(normal_map, dim=0, p=2),
        normal_map,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity_map": opacity_map,
        "depth_map": depth_map,
        "normal_map_from_depth": normal_map_from_depth,
        "normal_from_depth_mask": normal_from_depth_mask,
        "normal_map": normal_map,
        "normal_mask": normal_mask,
        "albedo_map": albedo_map,
        "roughness_map": roughness_map,
        "metallic_map": metallic_map,
    }
    # rendered_image, radii = rasterizer(
    #     means3D=means3D,
    #     means2D=means2D,
    #     shs=shs,
    #     opacities=opacity,
    #     scales=scales,
    #     rotations=rotations,
    #     colors_precomp=rgb,
    # )
    # rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.
    # return {
    #     "render": rendered_image,
    #     "viewspace_points": screenspace_points,
    #     "visibility_filter" : radii > 0,
    #     "radii": radii,
    # }
