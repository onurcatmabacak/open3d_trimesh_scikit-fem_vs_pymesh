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


def open3d_wireframe(filename):

    mesh = load_mesh_open3d(filename) 
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    o3d.visualization.draw_geometries([wireframe])


def trimesh_to_open3d(filename, classification=True):

    # To load and clean up mesh - "remove vertices that share position"

    classification = False
    if classification:
        mesh_ = tri.load_mesh(filename, process=True)
        mesh_.remove_duplicate_faces()
    else:
        mesh_ = tri.load_mesh(filename, process=False)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(mesh_.faces)
        mesh.compute_vertex_normals()

    return mesh


def open3d_to_trimesh(filename):

    mesh = load_mesh_open3d(filename)
    triangle_normals = mesh.compute_triangle_normals() 
    vertex_normals = mesh.compute_vertex_normals()
    mesh = tri.Trimesh(vertices=mesh.vertices,faces=mesh.triangles, \
    triangle_normals=triangle_normals, vertex_normals=vertex_normals)

    return mesh

def tetrahedral_mesh(filename):

    mesh = load_mesh_open3d(filename)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(3000)
    tetramesh = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    
    return tetramesh




def trimesh_to_open3d(filename, classification=True):

    # To load and clean up mesh - "remove vertices that share position"

	classification = False
	if classification:
		mesh_ = tri.load_mesh(filename, process=True)
		mesh_.remove_duplicate_faces()
	else:
		mesh_ = tri.load_mesh(filename, process=False)
		mesh = o3d.geometry.TriangleMesh()
		mesh.vertices = o3d.utility.Vector3dVector(mesh_.vertices)
		mesh.triangles = o3d.utility.Vector3iVector(mesh_.faces)
		mesh.compute_vertex_normals()
		
		o3d.visualization.draw_geometries([mesh])
		
		return mesh
		
def trimesh_to_open3d_as_open(filename):

	mesh = tri.load_mesh(filename)
	print(tri.base.Trimesh.as_open3d(mesh))
		
def form_mesh_open3d(filename): 

	mesh_tri = tri.load_mesh(filename)
	mesh = o3d.geometry.TriangleMesh()
	mesh.vertices = o3d.utility.Vector3dVector(mesh_tri.vertices)
	mesh.triangles = o3d.utility.Vector3iVector(mesh_tri.faces)
	mesh.compute_vertex_normals()
	print(mesh.get_axis_aligned_bounding_box())
	#print(mesh.is_self_intersecting())

	mesh3d = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, 10.0)

	o3d.visualization.draw_geometries([mesh3d])
    
    
filename = "bunny.obj"
#save_mesh_open3d(filename)
#save_mesh_trimesh(filename)
#open3d_wireframe(filename)
#tetrahedral_mesh(filename)
#trimesh_to_open3d(filename, classification=True)
#delete_get_set_attributes_open3d(filename)
#trimesh_to_open3d_as_open(filename)
form_mesh_open3d(filename)    
    