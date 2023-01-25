import open3d as o3d
import trimesh as tri
import numpy as np
import scipy
import skfem
from skfem import *
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot
import networkx as nx
from math import *
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

def vertex_mean_curvature(mesh, radius):	
	
	#mesh_open3d = load_mesh_open3d(filename)
	#mesh = convert_open3d_to_trimesh(mesh_open3d)

	mean_curvature = tri.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius)/tri.curvature.sphere_ball_intersection(1, radius) 

	return mean_curvature

def get_smoothed(mesh, vertex_curvature):

	n = len(mesh.vertices)
	result = np.zeros(n)
	for j in range(n):
		r = vertex_curvature[j] + vertex_curvature[mesh.vertex_neighbors[j]].mean()
		r *= 0.5
		result[j] = r
	return result

def filter_small_radii(inds, mesh, percentage=0.9, filter=2):
	from neckpinch import get_all_neighbours # where is this function??

	#mesh.enable_connectivity()
	flags = np.zeros(len(mesh.vertices))
	inds0 = np.array(inds)
	flags[inds0] = 1
	result = []
	tot = 0
	for i in inds0:
		neighbours = get_all_neighbours(mesh, i, filter)
		tot += len(neighbours)
		# neighbours = mesh.get_vertex_adjacent_vertices(i)
		a = np.sum(flags[neighbours] / len(neighbours))
		if a > percentage:
			result.append(i)
	logging.info(f"#neighbours={tot / len(inds0)}")
	return result

def boundary_edges(mesh):

	#mesh_open3d = load_mesh_open3d(filename)
	#mesh = convert_open3d_to_trimesh(mesh_open3d)
	unique_edges = mesh.edges[tri.grouping.group_rows(mesh.edges_sorted, require_count=1)]

	return unique_edges

def get_small_radii(mesh, radius, filter=2, percentage=0.9):
	gauss_vertex = vertex_gaussian_curvature(mesh, radius)
	mean_vertex = vertex_mean_curvature(mesh, radius)
	g = gauss_vertex
	# g = get_smoothed(pm_mesh, g)
	h = mean_vertex * 2
	# h = get_smoothed(pm_mesh, h)
	e = h * h - 4 * g
	bd_edges = boundary_edges(mesh)
	bd_vertices = np.unique(bd_edges.ravel())
	n_bounds = len(bd_vertices)
	e[bd_vertices] = 0
	inds = np.where(e < 0)[0]
	# logging.info(f"{h[inds].max()},{g[inds].max()},{g[inds].min()}")
	# logging.info(f"e < 0 : #={len(inds)}")
	e[inds] = 0
	e = np.sqrt(e)
	k1 = 0.5 * (h + e)
	k2 = 0.5 * (h - e)

	ind_gauss_negative = np.where(g < 0)[0]
	negative = len(ind_gauss_negative) - n_bounds
	rest = len(g) - n_bounds
	#logging.info(f"percentage of negative curvature {negative / rest}")
	print(f"percentage of negative curvature {negative / rest}")
	# diff_gauss = np.abs(k1*k2 - g)[ind_gauss_negative]
	# print("diff_gauss", np.quantile(diff_gauss,.9), diff_gauss.mean(), diff_gauss.max(), diff_gauss.min())
	eps = 1e-5
	r1 = 1.0 / (eps + np.abs(k1))
	r2 = 1.0 / (eps + np.abs(k2))
	rmin = np.min((r1, r2), axis=0)
	rmax = np.max((r1, r2), axis=0)
	ind0 = np.where((rmin < radius) & (g < 0) & (rmin < 4 * rmax))[0]
	#if filter:
	#    ind0 = filter_small_radii(ind0, pm_mesh, filter=filter, percentage=percentage)
	return ind0

def not_hyperbolic(mesh):
	bd_edges = boundary_edges(mesh)
	bd_vertices = np.unique(bd_edges.ravel())

	points = mesh.bounding_box_oriented.sample_volume(count=1000)
	_, radius_all = tri.proximity.max_tangent_sphere(mesh, points)
	radius = np.max(radius_all)

	g = np.array(vertex_gaussian_curvature(mesh, radius))
	g[bd_vertices] = 0
	ind_gauss_negative = np.where(g > 1e-3)[0]
	return ind_gauss_negative

def form_mesh_trimesh(vertices, faces):
	return tri.base.Trimesh(vertices=vertices, faces=faces)

def normalize_mesh(vertices, faces):
	centroid = np.mean(vertices, axis=0)
	vertices -= centroid
	radii = norm(vertices, axis=1)
	vertices /= np.amax(radii)
	return form_mesh_trimesh(vertices, faces)


def print2outershell(what):
	sys.__stdout__.write(f"{what}\n")
	sys.__stdout__.flush()

def get_normalized_laplacian(adjacency_matrix):
	import scipy.sparse as spsp
	# Calculate diagonal matrix of node degrees.
	degree = spsp.dia_matrix((adjacency_matrix.sum(axis=0), np.array([0])), shape=adjacency_matrix.shape)
	degree = degree.tocsr()

	# Calculate sparse graph Laplacian.
	adjacency_matrix = spsp.csr_matrix(-adjacency_matrix + degree, dtype=np.float64)

	# Calculate inverse square root of diagonal matrix of node degrees.
	degree.data = np.real(1/np.sqrt(degree.data))

	# Calculate sparse normalized graph Laplacian.
	normalized_laplacian = degree*adjacency_matrix*degree

	return normalized_laplacian

def get_unnormalized_laplacian(adjacency_matrix):
	import scipy.sparse as spsp
	# Calculate diagonal matrix of node degrees.
	degree = spsp.dia_matrix((adjacency_matrix.sum(axis=0), np.array([0])), shape=adjacency_matrix.shape)
	degree = degree.tocsr()

	# Calculate sparse graph Laplacian.
	laplacian = spsp.csr_matrix(-adjacency_matrix + degree, dtype=np.float64)

	return laplacian 

def get_random_walk_laplacian(adjacency_matrix):
	import scipy.sparse as spsp
	# Calculate diagonal matrix of node degrees.
	degree = spsp.dia_matrix((adjacency_matrix.sum(axis=0), np.array([0])), shape=adjacency_matrix.shape)
	degree = degree.tocsr()

	# Calculate sparse graph Laplacian.
	adjacency_matrix = spsp.csr_matrix(-adjacency_matrix + degree, dtype=np.float64)

	# Calculate inverse of diagonal matrix of node degrees.
	degree.data = np.real(1/degree.data)

	# Calculate sparse normalized graph Laplacian.
	random_walk_laplacian = degree*adjacency_matrix

	return random_walk_laplacian 

def integrate(mesh, time_step, L):

	from scipy.sparse import linalg as sla
	from mindboggle.shapes.laplace_beltrami import fem_laplacian
	from mindboggle.shapes.laplace_beltrami import computeAB
	from mindboggle.guts.graph import graph_laplacian
	import networkx as nx

	print(mesh.vertices)
	vertices = mesh.vertices
	faces = mesh.faces

	graph = tri.graph.vertex_adjacency_graph(mesh)
	adjacency_matrix = nx.adjacency_matrix(graph)
	#L = get_normalized_laplacian(adjacency_matrix)
	L = get_random_walk_laplacian(adjacency_matrix)
	print(L)
	#quit()
	_, M = computeAB(vertices, faces)
	#L = fem_laplacian(vertices, faces, spectrum_size=mesh.vertices.shape[0]) #, normalization="area", verbose=False)
	#L = csgraph.laplacian(vertices)

	bbox_min, bbox_max = mesh.bounds
	s = np.amax(bbox_max - bbox_min)  # why?
	S = M + (time_step * s) * L
	
	lu = sla.splu(S)
	mv = M * mesh.vertices
	vertices = lu.solve(mv)

	return vertices, mesh.faces

# def get_vtk_curvature(mesh):
#     vtk_mesh = wtv.tvtk_curvatures.get_poly_data_surface_with_curvatures(mesh.vertices, mesh.faces)
#     scs = vtk_mesh.point_data.scalars
#     return np.array(scs)


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
#vertex_curvature = vertex_gaussian_curvature(mesh, 1.0)
#print( get_smoothed(mesh, vertex_curvature) )
#print( get_small_radii(mesh, radius=1.0, filter=2, percentage=0.9) )
#print( not_hyperbolic(mesh) )

print( integrate(mesh, time_step=0.1, L=0.1) )

