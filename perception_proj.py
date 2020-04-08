#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
import pcl
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
  #  print('   yaml output dict_list', dict_list)
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = .005
    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    # TODO: PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()

     # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    print('pass through filter in z', axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    #PassThrough Filter side to side too
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max = 0.4
    passthrough.set_filter_limits(axis_min, axis_max)
    print('pass through filter in y', axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    
    # TODO: RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    print('RANSAC plane')

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = .008
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    # Extract inliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)

    # Extract outliers
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    
     #apply k-means filtering to get noise out
    k_filtered = cloud_objects.make_statistical_outlier_filter()
    k_filtered.set_mean_k(3)
    k_filtered.set_std_dev_mul_thresh(0.01)
    cloud_objects = k_filtered.filter()
    

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)  # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    min_clust = 20   
    max_clust = 2500
    clust_tol = 0.02
    ec.set_ClusterTolerance(clust_tol)
    ec.set_MinClusterSize(min_clust)
    ec.set_MaxClusterSize(max_clust)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    print('cluster from ',min_clust, max_clust)
    

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    print('cluster #', len(cluster_indices),' with cluster bounds [', min_clust, max_clust,']')
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        print('cluster ', j, ' size: ', len(cluster_indices[j]))
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    print('pcl converted to ros')

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)
    print('ros msgs published')

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    nbins = 90
    using_hsv = True

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)
        
        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, nbins, using_hsv)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals, nbins)
        feature = np.concatenate((chists, nhists))
        
        print('histograms computed')

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .3 # was .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
        print('append detected object: ', label)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)
    
    
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    if len(detected_objects) > 0:
        try:
            print('sending to pr2 mover # obj', len(detected_objects))
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
    else: 
        print("no objectd detected")
    return

    
def pr2_mover(object_list):
   # print(object_list) #what is coming in? 
    print(len(object_list), ' objects coming into pr2mover fn')
        

    # TODO: Initialize variables
    #these are requested (p71)
    test_scene_num = Int32() #std_msgs
    test_scene_num.data = 3 #see p 72
    object_name = String() #std_msgs / p71,72
    pick_pose = Pose() #geometry_msgs / p 72
    #others that i'm not doing
    arm_name = String()
    place_pose = Pose()
    dict_list = [] #p73

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list') #from p 70
 #   print('object_list sent to pr2_mover', object_list)
 #   print('object_list_param', object_list_param)

    # TODO: Parse parameters into individual variables
    

    # TODO: Rotate PR2 in place to capture side tables for the collision map
    
    #initialize next two lines for centroids (p70)
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    for object in object_list: #match to rospy get param
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])
        print('   calculated centroid of object: ', object.label)
        

    # TODO: Loop through the pick list
    for i in range(0, len(object_list_param)):
  #  for i in range(0, len(object_list)):
        object_name.data = object_list_param[i]['name'] #from p 70
        object_group = object_list_param[i]['group'] #from p70
  #      print('object_group: ', object_group)
        
        #need to put centroids into pick pose data
        pick_pose.position.x = np.asscalar(centroids[i][0])
        pick_pose.position.y = np.asscalar(centroids[i][1])
        pick_pose.position.z = np.asscalar(centroids[i][2])
    
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        
        # TODO: Create 'place_pose' for the object
        #done below

        # TODO: Assign the arm to be used for pick_place
        #green is on right, red is on left (as shown in Gazebo)
        if object_group == 'green':
            arm_name.data = 'right'
            #green on right
            place_pose.position.x = 0
            place_pose.position.y = -0.71
            place_pose.position.z = 0.605
        elif object_group == 'red':
            arm_name.data = 'left'
            #red on left
            place_pose.position.x = 0
            place_pose.position.y = 0.71
            place_pose.position.z = 0.605
        
   #     arm_name.data = 'not 

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        #following from p73
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)
        
        

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, arm_name, object_name, pick_pose, place_pose) #this was here, updated names being passed

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    #send output yaml filename and dictionary using send yaml fn
    yaml_name = 'output_'+str(test_scene_num.data)+'.yaml'
    send_to_yaml(yaml_name, dict_list)
    print(yaml_name, '... yaml file sent')
    
    return


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size = 1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size = 1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size = 1)
    #fancy stuff
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
