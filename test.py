import open3d as o3d
import trimesh as tri
import numpy as np
import scipy
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

	"""
	Find and fix problems with self.face_normals and self.faces winding direction.

	For face normals ensure that vectors are consistently pointed outwards, and that self.faces is wound in the correct direction for all connected components.

	Parameters
	multibody (None or bool) – Fix normals across multiple bodies if None automatically pick from body_count
	"""
	hull.fix_normals()
	hullStats["is_oriented"] = True #face normal orientation is fixed

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

def vertex_gaussian_curvature(mesh, radius=None):

	#mesh_open3d = load_mesh_open3d(filename)
	#mesh = convert_open3d_to_trimesh(mesh_open3d)

	if radius == None:
		points = mesh.bounding_box_oriented.sample_volume(count=1000)
		_, radius_all = tri.proximity.max_tangent_sphere(mesh, points)
		radius = np.max(radius_all)

	gaussian_curvature = tri.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius)/tri.curvature.sphere_ball_intersection(1, radius)
	
	#print(gauss)		
	return gaussian_curvature

def vertex_mean_curvature(mesh, radius=None):	
	
	#mesh_open3d = load_mesh_open3d(filename)
	#mesh = convert_open3d_to_trimesh(mesh_open3d)

	if radius == None:
		points = mesh.bounding_box_oriented.sample_volume(count=1000)
		_, radius_all = tri.proximity.max_tangent_sphere(mesh, points)
		radius = np.max(radius_all)

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

	g = np.array(vertex_gaussian_curvature(mesh))
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

def print_out_stat(what, ar):
	s = "{} mean={} max={} >80%={} >99%={}".format(
		what, np.mean(ar), np.max(ar), np.quantile(ar, 0.8), np.quantile(ar, 0.99)
	)
	logging.info(s)


def _modified_mean_curvature_flow(mesh, L, num_itrs=1, time_step=1e-3):
	bd_edges = boundary_edges(mesh)
	bd_vertices = np.unique(bd_edges.ravel())
	mesh = form_mesh_trimesh(mesh.vertices, mesh.faces)

	v0 = mesh.vertices

	bd0 = v0[bd_vertices]
	result = [mesh]
	for i in range(num_itrs):
		try:
			# curvature = get_vtk_curvature(mesh)
			# curvature[bd_vertices] = 0
			# print_out_stat("mean curvature", np.abs(curvature))
			verts, faces = integrate(mesh, time_step, L)  # Keep Laplacian constant to avoid neck pinches
			# print(np.mean(dv), np.max(dv), np.quantile(dv,.8))
			verts[bd_vertices] = bd0
			dv = np.abs(verts - v0)
			print_out_stat("vertex_diff", dv)
			mesh = form_mesh_trimesh(mesh.vertices, mesh.faces)
			result.append(mesh)
			v0 = mesh.vertices

		except Exception as e:
			# error with ldlt - try again but this time with sparseLU. Does it cause seg-fault?
			logging.error("Mean curvature flow failed at iteration %d with error %s", i, str(e))
			time_step /= 2
			if time_step < 1e-6:
				return mesh
			continue
	return result

def is_valid_mesh(mesh):
	# bounds = mesh.boundary_vertices
	# if len(bounds) > 0:
	#    logging.info("mesh has boundary")
	#    return False
	ints = np.asarray(mesh.as_open3d.get_self_intersecting_triangles()) 
	return mesh.as_open3d.is_edge_manifold() and mesh.as_open3d.is_vertex_manifold() and len(ints) == 0

def get_narrow_tunnels(mesh, min_dist: float, tree=None):

	centroids = mesh.triangles_center
	normals = mesh.face_normal

	neck_pinch_faces = {}

	if not tree:
		tree = scipy.spatial.cKDTree(centroids, leafsize=11)

	inds = tree.query_ball_point(centroids, min_dist, workers=-1)

	n = len(inds)

	for j in range(n):

		i_normals = normals[inds[j]]
		dots = np.dot(i_normals, i_normals[0])
		opposite = np.argmin(dots)
		
		if dots[opposite] < -0.9:
		
			already = None
		
			for k in inds[j]:
				if k in neck_pinch_faces:
					already = neck_pinch_faces[k]

			if already is None:
				already = j

			for k in inds[j]:
				neck_pinch_faces[k] = already

	clusters = defaultdict(list)

	for i, k in neck_pinch_faces.items():
		clusters[k].append(i)

	return clusters, tree


def get_narrow_tunnel_vertices(mesh, min_dist: float, tree=None):
    clusters, tree = get_narrow_tunnels(mesh, min_dist, tree)
    result = {}
    for k in clusters:
        verts = []
        for face_index in clusters[k]:
            verts.extend(mesh.faces[face_index])
        verts = list(set(verts))
        result[k] = verts
    return result, tree


def mesh2mesh_distance(mesh, min_dist: float):

	centroids = mesh.triangles_center

	clusters, tree = get_narrow_tunnels(mesh, min_dist)
	points = []

	logging.info(f"#clusters = {len(clusters)} with width < {min_dist}")

	for k in clusters:
		# print(k, clusters[k])
		points.extend(centroids[clusters[k]])

	return len(clusters), points


def euler(mesh):
    np = len(mesh.vertices)
    nf = len(mesh.faces)
    ne = (3 * nf + len(boundary_edges(mesh))) / 2
    euler = np + nf - ne
    return euler

def genus(mesh):
    return 1 - euler(mesh) / 2


def mesh_statistics(mesh, calc_genus=True, distances=[]):
	# outdated statistics
	# tr = context_manager.ResultsManager()

	# bd_edges = boundary_edges(mesh)
	# bd_vertices = np.unique(bd_edges.ravel())

	def zero_nan(what, replace_value=0):
		what[~np.isfinite(what)] = replace_value

	mesh.area_faces
	gauss = vertex_gaussian_curvature(mesh)
	# tr.area_of_smooth_contour = np.sum(mesh.area_faces)

	zero_nan(gauss)
	bd_edges = boundary_edges(mesh)
	bd_vertices = np.unique(bd_edges.ravel())
	gauss[bd_vertices] = 0

	# tr.gaussian_av = np.mean(gauss)
	mean = vertex_mean_curvature(mesh)
	mean[bd_vertices] = 0
	zero_nan(mean)
	mc = np.abs(mean)
	# tr.absmean_av = np.mean(mc)  # /result["gaussian_integral"]
	# tr.absmean_q0p9 = np.quantile(mc, .9)
	# tr.absmean_q0p5 = np.quantile(mc, .5)
	# if calc_genus:
	# tr.genus = genus(mesh)
	# for r in distances:
	#     anz, points = mesh2mesh_distance(mesh, r)
	#     result[f"anz_narrow_clusters_{r}"] = anz

	gauss_vertex = vertex_gaussian_curvature(mesh)
	g = gauss_vertex
	bd_edges = boundary_edges(mesh)
	bd_vertices = np.unique(bd_edges.ravel())
	n_bounds = len(bd_vertices)
	ind_gauss_negative = np.where(g < 0)[0]
	negative = len(ind_gauss_negative) - n_bounds
	rest = len(g) - n_bounds
	# tr.percentage_of_negative_curvature = negative / rest
	# logging.info(f"percentage of negative curvature {tr.percentage_of_negative_curvature}")


def test_mesh_integrity(mesh):
	'''should be renamed to test_surface'''
	mesh_statistics(mesh)
	tr = context_manager.ResultsManager()

	mesh.fix_normals()
	tr.validation_tests_stats.subtest_self_intersection = np.asarray(mesh.as_open3d.get_self_intersecting_triangles())
	tr.validation_tests_stats.subtest_is_vertex_manifold = mesh.as_open3d.is_vertex_manifold()
	tr.validation_tests_stats.subtest_is_edge_manifold = mesh.as_open3d.is_edge_manifold()
	tr.validation_tests_stats.subtest_is_oriented = True


def test_mesh_printability(mesh, case_config):

    test_mesh_integrity(mesh)
    
	tr = context_manager.ResultsManager()
    
	min_dist = case_config.MIN_CHANNEL
    
	anz, points = mesh2mesh_distance(mesh, min_dist)
    
	tr.print_stats.anz_narrow_clusters = anz
    tr.print_stats.face_centroids_narrow = points
    
	good = True
    
	if len(tr.validation_tests_stats.subtest_self_intersection) > 0:
        good = False

    if not tr.validation_tests_stats.subtest_has_no_outer_component:
        good = False

    for val in (
        tr.validation_tests_stats.subtest_is_vertex_manifold,
        tr.validation_tests_stats.subtest_is_edge_manifold,
        tr.validation_tests_stats.subtest_is_oriented,
    ):
        if not val:
            good = False

    if (
        tr.validation_tests_stats.subtest_has_no_degenerated_triangle_angle
        or tr.validation_tests_stats.subtest_has_no_degenerated_triangle_area
    ):
        good = False
		
    if tr.print_stats.anz_narrow_clusters:
        good = False

    tr.print_stats.is_mesh_ok = good
    # radius = smoothing_tweaks.MIN_RADIUS_STOP_LAPLACIAN
    # result["_stop_smoothing_radius"] = mesh.vertices[get_small_radii(mesh, radius)]
    # radius = smoothing_tweaks.MIN_RADIUS_GROW
    # radius = 1
    # result["_grow_radius"] = mesh.vertices[get_small_radii(mesh, radius, percentage=.4, filter=3)]
    tr.print_stats.non_hyperbolic = mesh.vertices[not_hyperbolic(mesh)]


def split_solid_surface_along_surface(solid_surface, surface):
    '''
    Function splits the solid_surface into two parts:
        - faces and vertices that have normals pointing in approximatly the same direction as the surfaces are added to one new mesh
        - faces and vertices that have normals pointing in approximatly the opposite direction as the surface are added to another new mesh
    ONLY WORKS PROPERLY ON BOUNDARY IF SURFACE CUTS THROUGH SOLID SURFACE
    Args:
        solid_surface {PyMesh Mesh}: Printable "3d" triangulated mesh.
        surface {PyMesh Mesh}: "2d" triangulated mesh.

    Returns:
        Both new meshes
    '''
    # Compute face normals of meshes
    solid_surface.add_attribute(pymesh_constants.FACE_NORMAL)
    solid_surface.add_attribute(pymesh_constants.FACE_CIRCUMCENTER)

    surface.add_attribute(pymesh_constants.FACE_NORMAL)
    surface.add_attribute(pymesh_constants.FACE_CIRCUMCENTER)

    # Reshape to get dimensions: number of faces x 3
    solid_face_dimensions = np.shape(solid_surface.faces)

    solid_normals = np.reshape(solid_surface.get_attribute(pymesh_constants.FACE_NORMAL), solid_face_dimensions)
    plane_normals = np.reshape(surface.get_attribute(pymesh_constants.FACE_NORMAL), np.shape(surface.faces))
    solid_face_circumcenters = np.reshape(
        solid_surface.get_attribute(pymesh_constants.FACE_CIRCUMCENTER), solid_face_dimensions
    )
    solid_circumcenters = np.reshape(surface.get_attribute(pymesh_constants.FACE_CIRCUMCENTER), np.shape(surface.faces))

    squared_distances, face_indices, closest_points = pm.distance_to_mesh(surface, solid_face_circumcenters)

    top_side_faces = []
    bottom_side_faces = []
    for face in range(solid_face_dimensions[0]):
        vector_solid_surface_center_to_surface_center = (
            solid_face_circumcenters[face] - solid_circumcenters[face_indices[face]]
        )
        normal_scalar_product = (
            vector_solid_surface_center_to_surface_center
            / np.linalg.norm(vector_solid_surface_center_to_surface_center)
        ).dot(plane_normals[face_indices[face]])
        if normal_scalar_product > 0:
            top_side_faces.append(face)
        elif normal_scalar_product < 0:
            bottom_side_faces.append(face)
        else:
            # orthogonal
            pass

    # extract submesh. The 0 means that no faces other than the saved indices are extracted.
    top = pm.submesh(solid_surface, top_side_faces, 0)
    bot = pm.submesh(solid_surface, bottom_side_faces, 0)

    return bot, top


def compute_wall_thickness_at_sample_points(solid_surface, surface, sample_points):
    '''
    Args:
        solid_surface {PyMesh Mesh}: Printable "3d" triangulated mesh.
        surface {PyMesh Mesh}: "2d" triangulated mesh.
        sample_points: Numpy List of sample points. Sample points should lay on surface (e.g. vertices, midpoints, ...)
    Returns:
        Numpy List of dimensions: "Number of sample_points" x 1
    '''
    bot, top = split_solid_surface_along_surface(solid_surface, surface)

    # I guess it's up for discussion if the wall width should be measured orthogonally to the mid surface or shortest distance
    squared_distances_up, _, _ = pm.distance_to_mesh(top, sample_points)
    squared_distances_down, _, _ = pm.distance_to_mesh(bot, sample_points)

    distances_up = np.sqrt(squared_distances_up)
    distances_down = np.sqrt(squared_distances_down)

    return distances_down + distances_up


def test_wall_thickness(solid_surface, surface):
    sample_points = surface.vertices
    wall_thickness = compute_wall_thickness_at_sample_points(solid_surface, surface, sample_points)
    return wall_thickness



def form_mesh(vertices, faces):
    return pm.form_mesh(vertices, faces)


def form_mesh_with_voxels(vertices, faces, voxels):
    return pm.form_mesh(vertices, faces, voxels)


def remove_isolated_vertices(mesh):
    return pm.remove_isolated_vertices(mesh)


def remove_duplicated_faces(mesh, fins_only):
    return pm.remove_duplicated_faces(mesh, fins_only=fins_only)


def remove_duplicated_vertices(mesh, tol=1e-12, importance=None):
    return pm.remove_duplicated_vertices(mesh, tol=tol, importance=importance)


def collapse_short_edges(mesh, eps):
    return pm.collapse_short_edges(mesh, eps)


def remove_obtuse_triangles(mesh, max_angle):
    return pm.remove_obtuse_triangles(mesh, max_angle=max_angle)


def remove_degenerated_triangles(mesh, n):
    return pm.remove_degenerated_triangles(mesh, n)


def load_mesh(path):
    return pm.load_mesh(path)


def save_mesh(filepath, mesh):
    return pm.save_mesh(filepath, mesh)


def tetgen():
    return pm.tetgen()


def detect_self_intersection(mesh):
    return pm.detect_self_intersection(mesh)


def separate_mesh(mesh, connectivity_type="auto"):
    return pm.separate_mesh(mesh, connectivity_type="auto")


def pymesh_mesh():
    return pm.Mesh


def get_pymesh_stats(mesh):
    return context_manager.ResultsManager()

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

#print( integrate(mesh, time_step=0.1, L=0.1) )
#print(is_valid_mesh(mesh))
#get_narrow_tunnels(mesh, 1.0, tree=None)
