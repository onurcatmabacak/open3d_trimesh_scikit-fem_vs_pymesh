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
	
def get_boundary_face_indices(filename):

	mesh = load_mesh_trimesh(filename)

	index = tri.grouping.group_rows(mesh.edges_sorted, require_count=1)
	boundary_vertices = np.unique(mesh.edges[index].flatten())
	boundary_faces = mesh.faces[mesh.edges_face[index]]
	boundary_face_indices = np.where(np.in1d(mesh.faces, boundary_faces))[0]
	print(boundary_face_indices)

def get_edge_lengths(vertices, faces):
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]

    length1 = np.linalg.norm(v2 - v3, axis=1)
    length2 = np.linalg.norm(v1 - v3, axis=1)
    length3 = np.linalg.norm(v1 - v2, axis=1)

    return np.array([length1, length2, length3]).T

def get_signed_volume(filename):
	"""
	function to get the signed volume of  3D MESH. This is based on the following formula
	Args:
		mesh:

	Returns:

	"""
	mesh = load_mesh_trimesh(filename)
	face_areas = mesh.area_faces
	face_centroids = mesh.triangles_center.reshape((-1, 3))
	face_normals = mesh.face_normals.reshape((-1, 3))
	product_barycenter_normal = np.sum(face_centroids[:, 0] * face_normals[:, 0])  # , axis=1)
	return 1 / 6 * (np.sum(product_barycenter_normal * face_areas))

def get_interior_angles(vertices, faces):
    # numpy (n_triangles, 3) array of edge lengths
    edge_lengths = get_edge_lengths(vertices, faces)

    # Apply cosine law
    a1 = apply_cosine_law(edge_lengths[:, 0], edge_lengths[:, 1], edge_lengths[:, 2])
    a2 = apply_cosine_law(edge_lengths[:, 1], edge_lengths[:, 0], edge_lengths[:, 2])
    a3 = apply_cosine_law(edge_lengths[:, 2], edge_lengths[:, 0], edge_lengths[:, 1])

    return np.rad2deg(np.array([a1, a2, a3]).T)

def apply_cosine_law(a: np.array, b: np.array, c: np.array):
    """
    Args:
        a:
        b:
        c:

    Returns:

    """
    return np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

def getHullStats(hull):
	# convenience function to get hull stats
	hullStats = dict()
	geometry = dict()
	face_areas = hull.area_faces

	hullStats["is_closed"] = hull.is_watertight
	hullStats["is_oriented"] = True #problem her, hull.is_oriented()

	# y is height, x is width and z is depth
	bbox = hull.bounds
	print(bbox)
	geometry["width"] = bbox[1][0] - bbox[0][0]
	geometry["height"] = bbox[1][1] - bbox[0][1]
	geometry["depth"] = bbox[1][2] - bbox[0][2]
	geometry["area"] = np.sum(face_areas)
	geometry["volume"] = hull.volume
	geometry["centroid"] = 0.5 * (bbox[0] + bbox[1])
	hullStats["geometry"] = geometry

	return hullStats

def scaleHull(hull, scale):
	# provide scale with array of x,y,z scale.
	# Example: Scale factor 2 => [2,2,2]
	print(hull.voxelized)
	return tri.Trimesh(hull.vertices * np.array(scale), hull.faces) #, hull.voxelized)

def vertex_gaussian_curvature(mesh, radius):

	#mesh_open3d = load_mesh_open3d(filename)
	#mesh = convert_open3d_to_trimesh(mesh_open3d)

	gaussian_curvature = tri.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius)/tri.curvature.sphere_ball_intersection(1, radius)
	
	#print(gauss)		
	return gaussian_curvature

def get_smoothed(mesh, vertex_curvature):

	n = len(mesh.vertices)
	result = np.zeros(n)
	for j in range(n):
		r = vertex_curvature[j] + vertex_curvature[mesh.vertex_neighbors[j]].mean()
		r *= 0.5
		result[j] = r
	return result

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

#integrate(filename, time_step=0.5)

# functions in wpm pymesh utils
mesh = load_mesh_trimesh(filename)
#print( getHullStats(mesh.convex_hull) )
#print( scaleHull(mesh.convex_hull, 2.0) )
vertex_curvature = vertex_gaussian_curvature(mesh, 1.0)
print( get_smoothed(mesh, vertex_curvature) )