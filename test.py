import open3d as o3d
import trimesh as tri
import numpy as np
import scipy
import skfem
from skfem import *
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot

def load_mesh_trimesh(filename):

    return tri.load_mesh(filename)

def load_mesh_open3d(filename):

    return o3d.io.read_triangle_mesh(filename)

def save_mesh_open3d(filename):

    mesh = load_mesh_open3d(filename) 
    o3d.io.write_triangle_mesh("bunny_saved_open3d.obj", mesh)
    
def save_mesh_trimesh(filename):
    
    mesh = load_mesh_trimesh(filename) 
    trimesh.exchange.export.export_mesh(mesh, "bunny_saved_trimesh.obj",)


def open3d_wireframe(filename):

    mesh = load_mesh_open3d(filename) 
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    o3d.visualization.draw_geometries([wireframe])
    

def tetrahedral_mesh(filename):

    mesh = load_mesh_open3d(filename)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(10000)
    tetramesh = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    
    print(tetramesh)
    o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)
		
def trimesh_to_open3d(filename):

	mesh = tri.load_mesh(filename)
	print(mesh.as_open3d)
		
def form_mesh_open3d(filename): 

	mesh_tri = tri.load_mesh(filename)
	mesh = o3d.geometry.TriangleMesh()
	mesh.vertices = o3d.utility.Vector3dVector(mesh_tri.vertices)
	mesh.triangles = o3d.utility.Vector3iVector(mesh_tri.faces)
	mesh.compute_vertex_normals()
	mesh.compute_triangle_normals()
	print(mesh.get_axis_aligned_bounding_box())
	#print(mesh.is_self_intersecting())

	mesh3d = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, 10.0)

	o3d.visualization.draw_geometries([mesh3d])
    
def convert_open3d_to_trimesh(mesh_open3d):

	mesh_open3d.compute_vertex_normals()
	mesh_open3d.compute_triangle_normals()
	
	print( np.asarray(mesh_open3d.vertex_normals), np.asarray(mesh_open3d.triangle_normals) )
	
	mesh_tri = tri.base.Trimesh(vertices=np.asarray(mesh_open3d.vertices), faces=np.asarray(mesh_open3d.triangles), vertex_normals=np.asarray(mesh_open3d.vertex_normals), face_normals=np.asarray(mesh_open3d.triangle_normals))
	return mesh_tri
    
def triangle_surface_area(filename):

	mesh_open3d = load_mesh_open3d(filename)
	mesh_open3d.compute_vertex_normals()
	mesh_open3d.compute_triangle_normals()

	mesh_tri = convert_open3d_to_trimesh(mesh_open3d)
	
	return mesh_tri.area_faces
    
    
def face_centers(filename):

	mesh_open3d = load_mesh_open3d(filename)
	mesh = convert_open3d_to_trimesh(mesh_open3d)
	return mesh.triangles_center
    
    
def vertex_gaussian_curvature(filename, radius):

	mesh_open3d = load_mesh_open3d(filename)
	mesh = convert_open3d_to_trimesh(mesh_open3d)

	gauss = tri.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius)/tri.curvature.sphere_ball_intersection(1, radius)
	
	print(gauss)		
	return gauss

def vertex_mean_curvature(filename, radius):	
	
	mesh_open3d = load_mesh_open3d(filename)
	mesh = convert_open3d_to_trimesh(mesh_open3d)

	mean = tri.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius)/tri.curvature.sphere_ball_intersection(1, radius) 

	print(mean)

	return mean
	
def boundary_edges(filename):

	mesh_open3d = load_mesh_open3d(filename)
	mesh = convert_open3d_to_trimesh(mesh_open3d)
	unique_edges = mesh.edges[tri.grouping.group_rows(mesh.edges_sorted, require_count=1)]

	print(unique_edges)

def scikit_fem_triangle_rearrange(triangles):

	mimi = []

	for i in range(len(triangles[0])):

		mimi.append(triangles[:,i])
	
	return np.asarray(mimi)

def integrate(filename, time_step):

	mesh = MeshTri.load(filename)
	print(mesh.p.T)
	mesh.p = mesh.p.T
	mesh.t = mesh.t.T
	#mesh_open3d = load_mesh_open3d(filename)
	#print(mesh.p, scikit_fem_triangle_rearrange(mesh.t))

	element = {'u': ElementVector(ElementTriP2()),
			'p': ElementTriP1()}
	basis = {variable: Basis(mesh, e, intorder=3)
			for variable, e in element.items()}

	A = asm(vector_laplace, basis['u'])
	B = asm(divergence, basis['u'], basis['p'])
	C = asm(mass, basis['p'])

	print( C )

	quit()
	

filename = "bunny.obj"
#save_mesh_open3d(filename)
#save_mesh_trimesh(filename)
#open3d_wireframe(filename)
#tetrahedral_mesh(filename)
#trimesh_to_open3d(filename, classification=True)
#delete_get_set_attributes_open3d(filename)
#trimesh_to_open3d_as_open(filename)
#form_mesh_open3d(filename)

#triangle_surface_area(filename)
#trimesh_to_open3d_as_open(filename)
#face_centers(filename)

#vertex_gaussian_curvature(filename, 0.05)
#vertex_mean_curvature(filename, 0.05)

#boundary_edges(filename)

integrate(filename, time_step=0.5)
