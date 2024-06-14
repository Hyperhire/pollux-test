import open3d as o3d

# Load your mesh
input_path = "output/test608/white_background_ori_232_6ceabe18-6pad12_originalgtmask_scaleloss_1/poisson_mesh_10"

mesh = o3d.io.read_triangle_mesh(input_path + '_pruned.ply')

# Simplify the mesh to a target number of triangles
target_number_of_triangles = 1000000
simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles)
o3d.io.write_triangle_mesh(input_path + '_simple_filter.ply', simplified_mesh)

# Apply simple Laplacian smoothing
number_of_iterations = 3
smoothed_mesh = simplified_mesh.filter_smooth_simple(number_of_iterations)
o3d.io.write_triangle_mesh(input_path + '_simple_laplacian.ply', smoothed_mesh)

# Remove duplicated vertices and triangles
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
o3d.io.write_triangle_mesh(input_path + '_simple_v_and_t.ply', smoothed_mesh)

print("Mesh optimization complete. Optimized mesh saved to 'path/to/your/cleaned_mesh.ply'.")
