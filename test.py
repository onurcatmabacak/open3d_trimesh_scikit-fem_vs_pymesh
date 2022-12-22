import open3d as o3d
import trimesh as tri
import numpy as np
import skfem
import scipy

def load_mesh_trimesh(filename):

    return tri.load_mesh(filename)


def load_mesh_open3d(filename):

    return o3d.io.read_triangle_mesh(filename)


def save_mesh_trimesh(filename):

    mesh = load_mesh_trimesh(filename) 
    trimesh.exchange.export.export_mesh(mesh, "bunny_saved_trimesh.obj",)


def save_mesh_open3d(filename):

    mesh = load_mesh_open3d(filename) 
    o3d.io.write_triangle_mesh(mesh, "bunny_saved_open3d.obj")


filename = "bunny.obj"
save_mesh_open3d(filename)
save_mesh_trimesh(filename)