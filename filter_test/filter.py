import open3d as o3d

# Load your mesh
input_path = "output/test608/white_background_ori_232_6ceabe18-6pad12_originalgtmask_scaleloss_1/poisson_mesh_10"

mesh = o3d.io.read_triangle_mesh(input_path + '_pruned.ply')

# Simplify the mesh to a target number of triangles
target_number_of_triangles = 900000
simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles)
o3d.io.write_triangle_mesh(input_path + '_sqd900k.ply', simplified_mesh)

# Apply bilateral smoothing
taubin_mesh = mesh.filter_smooth_taubin(number_of_iterations=10, mu=0.5)
o3d.io.write_triangle_mesh(input_path + '_taubin.ply', taubin_mesh)

# Apply simple Laplacian smoothing
number_of_iterations = 5
laplacian_mesh = mesh.filter_smooth_simple(number_of_iterations)
o3d.io.write_triangle_mesh(input_path + '_laplacian.ply', laplacian_mesh)

# Remove duplicated vertices and triangles
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
o3d.io.write_triangle_mesh(input_path + '_simple_v_and_t.ply', mesh)

print("Mesh optimization complete. Optimized mesh saved.")
