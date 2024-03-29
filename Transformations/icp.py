# Written by Open3D


from PIL import ImageTk, Image
import cv2
import os
import torch
import copy
from os.path import splitext
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


voxel_size = 0.05  # means 5cm for the datase

def icp(pcd, pcd2, Matrix):

    def draw_registration_resultold(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        #source_temp.paint_uniform_color([1, 0.706, 0])
        #target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    trans_initold = np.asarray([[0.862, 0.011, -0.507, 0],
                         [-0.139, 0.967, -0.215, 0],
                         [0.487, 0.255, 0.835, 0], [0.0, 0.0, 0.0, 1.0]])

    #draw_registration_result(pcd, pcd2, Matrix2)
    # 
    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        #source_temp.paint_uniform_color([1, 0.706, 0])
        #target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])
    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        o3d.geometry.estimate_normals(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    
    def prepare_dataset(voxel_size):
        print(":: Load two point clouds and disturb initial pose.")
        source = pcd
        target = pcd2
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        source.transform(trans_init)
        draw_registration_result(source, target, np.identity(4))
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh
        
    def execute_global_registration(source_down, target_down, source_fpfh,target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
        return result
        
    def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
        return result

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])  
    threshold = 0.02
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd, pcd2, threshold, Matrix, 
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)

    
    #source, target, source_down, target_down, source_fpfh, target_fpfh = \
    #prepare_dataset(voxel_size)
    #result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
    #print(result_ransac)
    #draw_registration_result(source_down, target_down, result_ransac.transformation)
    #result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
    #print(result_icp)
    draw_registration_result(pcd, pcd2, reg_p2p.transformation)