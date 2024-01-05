import torch
import trimesh
import numpy as np
import pysdf
from sdf.network_tcnn import SDFNetwork


STD_DEV = 1e-2
NUM_POINTS = 2**18


def calculate_f1(gt, pred):
    TP = torch.sum((gt == 1) & (pred == 1)).item()
    FN = torch.sum((gt == 1) & (pred == 0)).item()
    FP = torch.sum((gt == 0) & (pred == 1)).item()
    # recall
    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)
    # precision
    if TP + FP == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)
    # f1
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def read_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model = SDFNetwork()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def read_mesh(mesh_path):
    mesh = trimesh.load_mesh(mesh_path)
    vs = mesh.vertices
    vmin = vs.min(0)
    vmax = vs.max(0)
    v_center = (vmin + vmax) / 2
    v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
    vs = (vs - v_center[None, :]) * v_scale
    mesh.vertices = vs
    return mesh

def sample_boundary_points(mesh, std_dev=STD_DEV, num_points=NUM_POINTS):
    points = mesh.sample(num_points)
    points = np.array(points) + np.random.normal(0, std_dev, points.shape)
    return points

def sample_uniform_volume_points(num_points=NUM_POINTS):
    points = np.random.rand(num_points, 3) * 2 - 1
    return points

def calculate_occupancy_f1(mesh, model, points):
    sdf_fn = pysdf.SDF(mesh.vertices, mesh.faces)
    gt_sdfs = -sdf_fn(points)[:,None].astype(np.float32)
    gt = torch.tensor(gt_sdfs <= 0).cuda().to(torch.float32)
    points = torch.tensor(points).cuda()

    pred_sdfs = model(torch.tensor(points))
    pred = (pred_sdfs <= 0).to(torch.float32)
    average_f1 = (calculate_f1(gt, pred) + calculate_f1(1-gt, 1-pred))/2
    return average_f1

if __name__ == "__main__":
    num_samples = 50
    total_boundary_points_occupancy_f1 = 0
    total_uniform_volume_points_occupancy_f1 = 0
    for sample in range(num_samples):
        ckpt_path = f"test_out/{sample}/checkpoints/ngp.pth.tar"
        mesh_path = f"test_task_meshes/{sample}.obj"
        model = read_model(ckpt_path)
        mesh = read_mesh(mesh_path)
        boundary_points = sample_boundary_points(mesh)
        uniform_volume_points = sample_uniform_volume_points()
        boundary_points_occupancy_f1 = calculate_occupancy_f1(mesh, model, boundary_points)
        uniform_volume_points_occupancy_f1 = calculate_occupancy_f1(mesh, model, uniform_volume_points)
        total_boundary_points_occupancy_f1 += boundary_points_occupancy_f1
        total_uniform_volume_points_occupancy_f1 += uniform_volume_points_occupancy_f1
        print(f"Boundary points occupancy f1 for {sample} sample: {boundary_points_occupancy_f1}")
        print(f"Uniform volume points occupancy f1 for {sample} sample: {uniform_volume_points_occupancy_f1}\n")
        with open("results.txt", "a") as file:
            file.write(f"{sample} {boundary_points_occupancy_f1} {uniform_volume_points_occupancy_f1}\n")
    print(f"Average boundary points occupancy f1: {total_boundary_points_occupancy_f1 / num_samples}")
    print(f"Average uniform volume points occupancy f1: {total_uniform_volume_points_occupancy_f1 / num_samples}")
    with open("results.txt", "a") as file:
            file.write(f"Average {total_boundary_points_occupancy_f1 / num_samples} {total_uniform_volume_points_occupancy_f1 / num_samples}\n")