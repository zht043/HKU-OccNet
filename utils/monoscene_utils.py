import numpy as np
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

"""
Most of the code is taken from https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py

@inproceedings{zeng20163dmatch,
    title={3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions},
    author={Zeng, Andy and Song, Shuran and Nie{\ss}ner, Matthias and Fisher, Matthew and Xiao, Jianxiong and Funkhouser, Thomas},
    booktitle={CVPR},
    year={2017}
}
"""

import numpy as np

from numba import njit, prange
from skimage import measure

FUSION_GPU_MODE = 0


class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, vol_bnds, voxel_size, use_gpu=True):
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
        # self._trunc_margin = 10  # truncation on SDF
        self._color_const = 256 * 256

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)

        print(
            "Voxel volume size: {} x {} x {} - # points: {:,}".format(
                self._vol_dim[0],
                self._vol_dim[1],
                self._vol_dim[2],
                self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2],
            )
        )

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        self.gpu_mode = use_gpu and FUSION_GPU_MODE

        # Copy voxel volumes to GPU
        if self.gpu_mode:
            self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
            cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
            self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
            cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
            self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
            cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

            # Cuda kernel function (C++)
            self._cuda_src_mod = SourceModule(
                """
        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          // World coordinates to camera coordinates
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          // Camera coordinates to image pixels
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          // Skip if outside view frustum
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;
          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;
          // Integrate TSDF
          float trunc_margin = other_params[4];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
          weight_vol[voxel_idx] = w_new;
          tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
          // Integrate color
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
        }"""
            )

            self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

            # Determine block/grid size on GPU
            gpu_dev = cuda.Device(0)
            self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
            n_blocks = int(
                np.ceil(
                    float(np.prod(self._vol_dim))
                    / float(self._max_gpu_threads_per_block)
                )
            )
            grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
            grid_dim_y = min(
                gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x)))
            )
            grid_dim_z = min(
                gpu_dev.MAX_GRID_DIM_Z,
                int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))),
            )
            self._max_gpu_grid_dim = np.array(
                [grid_dim_x, grid_dim_y, grid_dim_z]
            ).astype(int)
            self._n_gpu_loops = int(
                np.ceil(
                    float(np.prod(self._vol_dim))
                    / float(
                        np.prod(self._max_gpu_grid_dim)
                        * self._max_gpu_threads_per_block
                    )
                )
            )

        else:
            # Get voxel grid coordinates
            xv, yv, zv = np.meshgrid(
                range(self._vol_dim[0]),
                range(self._vol_dim[1]),
                range(self._vol_dim[2]),
                indexing="ij",
            )
            self.vox_coords = (
                np.concatenate(
                    [xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0
                )
                .astype(int)
                .T
            )

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size, offsets=(0.5, 0.5, 0.5)):
        """Convert voxel grid coordinates to world coordinates."""
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        #    print(np.min(vox_coords))
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)

        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = (
                    vol_origin[j]
                    + (vox_size * vox_coords[i, j])
                    + vox_size * offsets[j]
                )
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume."""
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        # print(tsdf_vol.shape)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.0):
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
        """
        im_h, im_w = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[..., 2] * self._color_const
            + color_im[..., 1] * 256
            + color_im[..., 0]
        )

        if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
            for gpu_loop_idx in range(self._n_gpu_loops):
                self._cuda_integrate(
                    self._tsdf_vol_gpu,
                    self._weight_vol_gpu,
                    self._color_vol_gpu,
                    cuda.InOut(self._vol_dim.astype(np.float32)),
                    cuda.InOut(self._vol_origin.astype(np.float32)),
                    cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                    cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                    cuda.InOut(
                        np.asarray(
                            [
                                gpu_loop_idx,
                                self._voxel_size,
                                im_h,
                                im_w,
                                self._trunc_margin,
                                obs_weight,
                            ],
                            np.float32,
                        )
                    ),
                    cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                    cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                    block=(self._max_gpu_threads_per_block, 1, 1),
                    grid=(
                        int(self._max_gpu_grid_dim[0]),
                        int(self._max_gpu_grid_dim[1]),
                        int(self._max_gpu_grid_dim[2]),
                    ),
                )
        else:  # CPU mode: integrate voxel volume (vectorized implementation)
            # Convert voxel grid coordinates to pixel coordinates
            cam_pts = self.vox2world(
                self._vol_origin, self.vox_coords, self._voxel_size
            )
            cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
            pix_z = cam_pts[:, 2]
            pix = self.cam2pix(cam_pts, cam_intr)
            pix_x, pix_y = pix[:, 0], pix[:, 1]

            # Eliminate pixels outside view frustum
            valid_pix = np.logical_and(
                pix_x >= 0,
                np.logical_and(
                    pix_x < im_w,
                    np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0)),
                ),
            )
            depth_val = np.zeros(pix_x.shape)
            depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

            # Integrate TSDF
            depth_diff = depth_val - pix_z

            valid_pts = np.logical_and(depth_val > 0, depth_diff >= -10)
            dist = depth_diff

            valid_vox_x = self.vox_coords[valid_pts, 0]
            valid_vox_y = self.vox_coords[valid_pts, 1]
            valid_vox_z = self.vox_coords[valid_pts, 2]
            w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = self.integrate_tsdf(
                tsdf_vals, valid_dist, w_old, obs_weight
            )
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Integrate color
            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(
                255.0, np.round((w_old * old_b + obs_weight * new_b) / w_new)
            )
            new_g = np.minimum(
                255.0, np.round((w_old * old_g + obs_weight * new_g) / w_new)
            )
            new_r = np.minimum(
                255.0, np.round((w_old * old_r + obs_weight * new_r) / w_new)
            )
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = (
                new_b * self._color_const + new_g * 256 + new_r
            )

    def get_volume(self):
        if self.gpu_mode:
            cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
            cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = (
            verts * self._voxel_size + self._vol_origin
        )  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image"""
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array(
        [
            (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[0, 0],
            (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[1, 1],
            np.array([0, max_depth, max_depth, max_depth, max_depth]),
        ]
    )
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file."""
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                norms[i, 0],
                norms[i, 1],
                norms[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file."""
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n"
            % (
                xyz[i, 0],
                xyz[i, 1],
                xyz[i, 2],
                rgb[i, 0],
                rgb[i, 1],
                rgb[i, 2],
            )
        )



def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out


def vox2pix(cam_E, cam_k, 
            vox_origin, voxel_size, 
            img_W, img_H, 
            scene_size):
    """
    compute the 2D projection of voxels centroids
    
    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2
    
    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV 
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < img_W,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_H,
                pix_z > 0))))


    return torch.from_numpy(projected_pix), torch.from_numpy(fov_mask), torch.from_numpy(pix_z)



def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    sensor_pose = 10
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid

def get_projections(img_W, img_H, calib):
    scale_3ds = [1, 2]
    data = {}
    for scale_3d in scale_3ds:
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = 0.2
        
    
        cam_k = calib["P2"][:3, :3]
        T_velo_2_cam = calib["Tr"]
        
        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(
            T_velo_2_cam,
            cam_k,
            vox_origin,
            voxel_size * scale_3d,
            img_W,
            img_H,
            scene_size,
        )            

        data["projected_pix_{}".format(scale_3d)] = projected_pix
        data["pix_z_{}".format(scale_3d)] = pix_z
        data["fov_mask_{}".format(scale_3d)] = fov_mask 
    return data


def majority_pooling(grid, k_size=2):
    result = np.zeros(
        (grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size)
    )
    for xx in range(0, int(np.floor(grid.shape[0] / k_size))):
        for yy in range(0, int(np.floor(grid.shape[1] / k_size))):
            for zz in range(0, int(np.floor(grid.shape[2] / k_size))):

                sub_m = grid[
                    (xx * k_size) : (xx * k_size) + k_size,
                    (yy * k_size) : (yy * k_size) + k_size,
                    (zz * k_size) : (zz * k_size) + k_size,
                ]
                unique, counts = np.unique(sub_m, return_counts=True)
                if True in ((unique != 0) & (unique != 255)):
                    # Remove counts with 0 and 255
                    counts = counts[((unique != 0) & (unique != 255))]
                    unique = unique[((unique != 0) & (unique != 255))]
                else:
                    if True in (unique == 0):
                        counts = counts[(unique != 255)]
                        unique = unique[(unique != 255)]
                value = unique[np.argmax(counts)]
                result[xx, yy, zz] = value
    return result


def draw(
    voxels,
    # T_velo_2_cam,
    # vox_origin,
    fov_mask,
    # img_size,
    # f,
    voxel_size=0.4,
    # d=7,  # 7m - determine the size of the mesh representing the camera
):

    fov_mask = fov_mask.reshape(-1)
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )


    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255), :
    ]
    # print(np.unique(fov_voxels[:, 3], return_counts=True))
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255), :
    ]

    # figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    colors = np.array(
        [
            [0,0,0],
            [100, 150, 245],
            [100, 230, 245],
            [30, 60, 150],
            [80, 30, 180],
            [100, 80, 250],
            [255, 30, 30],
            [255, 40, 200],
            [150, 30, 90],
            [255, 0, 255],
            [255, 150, 255],
            [75, 0, 75],
            [175, 0, 75],
            [255, 200, 0],
            [255, 120, 50],
            [0, 175, 0],
            [135, 60, 0],
            [150, 240, 80],
            [255, 240, 150],
            [255, 0, 0],
        ]
    ).astype(np.uint8)
    colors_map = np.array(
        [
            [0,0,0],
            [100, 150, 245],
            [100, 230, 245],
            [30, 60, 150],
            [80, 30, 180],
            [100, 80, 250],
            [255, 30, 30],
            [255, 40, 200],
            [150, 30, 90],
            [255, 0, 255],
            [255, 150, 255],
            [75, 0, 75],
            [175, 0, 75],
            [255, 200, 0],
            [255, 120, 50],
            [0, 175, 0],
            [135, 60, 0],
            [150, 240, 80],
            [255, 240, 150],
            [255, 0, 0],
        ]
    ).astype(np.uint8)

    pts_colors = [f'rgb({colors[int(i)][0]}, {colors[int(i)][1]}, {colors[int(i)][2]})' for i in fov_voxels[:, 3]]
    out_fov_colors = [f'rgb({colors[int(i)][0]//3*2}, {colors[int(i)][1]//3*2}, {colors[int(i)][2]//3*2})' for i in outfov_voxels[:, 3]]
    pts_colors = pts_colors + out_fov_colors
    
    fov_voxels = np.concatenate([fov_voxels, outfov_voxels], axis=0)
    x = fov_voxels[:, 0].flatten()
    y = fov_voxels[:, 1].flatten()
    z = fov_voxels[:, 2].flatten()
    label = fov_voxels[:, 3].flatten()
    label_meanings= {
        0: "unlabeled",#
        1: "outlier",
        10: "car",#
        11: "bicycle",#
        13: "bus",
        15: "motorcycle",#
        16: "on-rails",
        18: "truck",#
        20: "other-vehicle",#
        30: "person",#
        31: "bicyclist",#
        32: "motorcyclist",#
        40: "road",#
        44: "parking",#
        48: "sidewalk",#
        49: "other-ground",#
        50: "building",#
        51: "fence",#
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",#
        71: "trunk",#
        72: "terrain",#
        80: "pole",#
        81: "traffic-sign",#
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle"
    }
    learning_map_inv= {
        0: 0,
        1: 10,
        2: 11,
        3: 15,
        4: 18,
        5: 20,
        6: 30,
        7: 31,
        8: 32,
        9: 40,
        10: 44,
        11: 48,
        12: 49,
        13: 50,
        14: 51,
        15: 70,
        16: 71,
        17: 72,
        18: 80,
        19: 81
    }
    
    fig = go.Figure()
    unique_tags = np.unique(label)
    for tag in unique_tags:
        mask = label == tag
        x_filtered, y_filtered, z_filtered = x[mask], y[mask], z[mask]
        fig.add_trace(go.Scatter3d(name = label_meanings[learning_map_inv[int(tag)]], x=x_filtered, y=y_filtered, z=z_filtered, mode='markers',
                                   marker=dict(
                                       size=2,
                                       color='rgb({},{},{})'.format(colors_map[int(tag)][0],colors_map[int(tag)][1],colors_map[int(tag)][2]),       
                                       opacity=1,
                                       symbol='square'
                                   )))
    '''
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers',
                    marker=dict(
                            size=2,
                            color=pts_colors,                # set color to an array/list of desired values
                            # colorscale='Viridis',   # choose a colorscale
                            opacity=1.0,
                            symbol='square'
                        ))])
    '''
    fig.update_layout(
    scene = dict(
        aspectmode='data',
        xaxis = dict(
            backgroundcolor="rgb(255, 255, 255)",
            gridcolor="black",
            showbackground=True,
            zerolinecolor="black",
            nticks=4, 
            visible=False,
            range=[-1,55],),
        yaxis = dict(
            backgroundcolor="rgb(255, 255, 255)",
            gridcolor="black",
            showbackground=True,
            zerolinecolor="black",
            visible=False,
            nticks=4, range=[-1,55],),
        zaxis = dict(
            backgroundcolor="rgb(255, 255, 255)",
            gridcolor="black",
            showbackground=True,
            zerolinecolor="black",
            visible=False,
            nticks=4, range=[-1,7],),
        bgcolor="black",
    ),
    width=800,
    height=600 
    )
    camera = dict(eye=dict(x=-2.3, y=0, z=1.5))
    fig.update_layout(scene_camera=camera)

    # fig = px.scatter_3d(
    #     fov_voxels, 
    #     x=fov_voxels[:, 0], y="y", z="z", color="label")
    # Draw occupied inside FOV voxels
    # plt_plot_fov = mlab.points3d(
    #     fov_voxels[:, 0],
    #     fov_voxels[:, 1],
    #     fov_voxels[:, 2],
    #     fov_voxels[:, 3],
    #     colormap="viridis",
    #     scale_factor=voxel_size - 0.05 * voxel_size,
    #     mode="cube",
    #     opacity=1.0,
    #     vmin=1,
    #     vmax=19,
    # )

    # # Draw occupied outside FOV voxels
    # plt_plot_outfov = mlab.points3d(
    #     outfov_voxels[:, 0],
    #     outfov_voxels[:, 1],
    #     outfov_voxels[:, 2],
    #     outfov_voxels[:, 3],
    #     colormap="viridis",
    #     scale_factor=voxel_size - 0.05 * voxel_size,
    #     mode="cube",
    #     opacity=1.0,
    #     vmin=1,
    #     vmax=19,
    # )

    

    # plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    # plt_plot_outfov.glyph.scale_mode = "scale_by_vector"

    # plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    # outfov_colors = colors
    # outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
    # plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors

    # mlab.show()
    fig.show()
    return fig





