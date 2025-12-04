import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

DATANAME = "shreycoin20.ply"
pcd = o3d.io.read_point_cloud("../RESULTS/" + DATANAME)

#PREPROCESSING

pcd_center = pcd.get_center()

#statistical outlier filter

nn = 16
std_multiplier = 10
filtered_pcd = pcd.remove_statistical_outlier(nn, std_multiplier)
outliers = pcd.select_by_index(filtered_pcd[1], invert = True)
filtered_pcd = filtered_pcd[0]
# o3d.visualization.draw_geometries([filtered_pcd, outliers])

#making voxel
print("Filtered PCD has", np.asarray(filtered_pcd.points).shape[0], "points")
print("Bounds (min, max):", filtered_pcd.get_min_bound(), filtered_pcd.get_max_bound())

# do down-sample
voxel_size = 0.000001  # your chosen value
pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size=voxel_size)

print("After voxel down-sample:", np.asarray(pcd_downsampled.points).shape[0], "points")
print("Downsampled bounds:", pcd_downsampled.get_min_bound(), pcd_downsampled.get_max_bound())

# visualize that result with draw_geometries
# o3d.visualization.draw_geometries([pcd_downsampled])

#Estimating Normals

nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())
radius_normals = nn_distance*4
pcd_downsampled.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normals, max_nn = 16), fast_normal_computation = True)
pcd_downsampled.paint_uniform_color([0.6,0.6,0.6])
# 

#Extracting data from open3d viewer

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 0.000277470622677356, 0.00028391377418301994, -0.00048709396360209213 ],
# 			"boundingbox_min" : 
# 			[
# 				-0.00028762751026079059,
# 				-0.0002811843587551267,
# 				-0.00099511316511780024
# 			],
# 			"field_of_view" : 60.0,
# 			"front" : [ 0.029275091137406489, 0.19988153010200582, 0.97938263358249189 ],
# 			"lookat" : 
# 			[
# 				-5.0784437917172909e-06,
# 				1.3647077139466188e-06,
# 				-0.00074110356435994615
# 			],
# 			"up" : [ 0.048825212105632086, 0.97834705975467806, -0.20112963315289129 ],
# 			"zoom" : 0.69166666666666721
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }

front = [ 0.029275091137406489, 0.19988153010200582, 0.97938263358249189 ]
up = [ 0.048825212105632086, 0.97834705975467806, -0.20112963315289129 ]
lookat = [-5.0784437917172909e-06,1.3647077139466188e-06,-0.00074110356435994615]
zoom =  0.69166666666666721

pcd = pcd_downsampled
# o3d.visualization.draw_geometries([pcd], zoom = zoom, up = up, lookat = lookat, front = front)

# RANSAC PLANAR SEGMENTATION

pt_to_plane_dist = 0.0005
plane_model, inliers = pcd.segment_plane(distance_threshold = pt_to_plane_dist, ransac_n = 3, num_iterations = 1000)
[a,b,c,d] = plane_model
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert = True)
inlier_cloud.paint_uniform_color([1.0,0.0,0.0])
outlier_cloud.paint_uniform_color([0.6,0.6,0.6])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom = zoom, up = up, lookat = lookat, front = front)

# #MultiOrder RANSAC
# max_plane_idx = 1 #number of shapes to extract
# segment_models = {}
# segments = {}

# rest = pcd
# for i in range(max_plane_idx):
#     colors = plt.get_cmap("tab20")(i)
#     segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=1000)
#     segments[i]=rest.select_by_index(inliers)
#     segments[i].paint_uniform_color(list(colors[:3]))
#     rest = rest.select_by_index(inliers, invert=True)
#     print("pass",i,"/",max_plane_idx,"done.")
# # o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest], zoom=zoom, front=front, lookat=lookat, up=up)
# #%X 7. DBSCAN sur rest
# labels = np.array(rest.cluster_dbscan(eps=0.5, min_points=5))
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")

# colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest], zoom=zoom, front=front, lookat=lookat, up=up)