import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import kalman

vid_path = "/Users/jsanchez/Software/gitprojects/mocap_studio/vnect_res"
vid_name = "vid1"

img_path = "%s/%s/images" % (vid_path, vid_name)
jnts_2d_path = "%s/%s/joints_2d" % (vid_path, vid_name)
jnts_3d_path = "%s/%s/joints_3d" % (vid_path, vid_name)

# num files in folder
num_files = len([name for name in os.listdir(img_path)])
print(num_files)
fps = 20

# vnect_links
vnect_links = [[0, 16], [16, 1], [1, 2], [1, 5], [2, 3], [3, 4], [4, 17], [5, 6],
[6, 7], [7, 18], [1,15], [15, 14], [14, 8], [8, 9], [9, 10], [10, 19], [14, 11],
[11, 12], [12, 13], [13, 20]]

avatar_links = [[7,6],[6,5],[5,4],[4,3],[3,2],[2,1],[1,0],[8,9],[9,10],[10,11],
[11,22],[12,13],[13,14],[14,15],[15,23],[0,16],[16,17],[17,18],[18,24],[0,19],
[19,20],[20,21],[21,25]]


def draw_joints_2s(test_img, joints, scale_x, scale_y):

    for joint in joints:
        # pt_x = int(joint[0] * scale_x)
        # pt_y = int(joint[1] * scale_y)
        pt_x = joint[0]
        pt_y = joint[1]

        cv2.circle(test_img, center=(pt_y, pt_x), radius=3, color=[0,0,255], thickness=-1)

    # Plot limb colors
    for link_idx, link in enumerate(vnect_links):

        x1 = joints[link[0], 0]
        y1 = joints[link[0], 1]
        x2 = joints[link[1], 0]
        y2 = joints[link[1], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 200 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 2),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = link_idx // 4
            # limb_color = map(lambda x: x + 35 * (link_idx % 4), [139, 53, 255])
            limb_color = [139, 53, 255]
            cv2.fillConvexPoly(test_img, polygon, color=limb_color)


def compute_avatar_links_length(skel_joints):

    mid_pt = (skel_joints[8] + skel_joints[12]) / 2
    nskel_jnts = np.append(skel_joints, [mid_pt], axis=0)

    link_segments = [[7,5],[5,26],[26,2],[2,0],[26,9],[9,10],[10,11],[11,22],
                     [26,13],[13,14],[14,15],[15,23],[0,16],[16,17],[17,18],
                     [18,24],[0,19],[19,20],[20,21],[21,25]]

    link_lengths = []
    for link_seg in link_segments:

        l1 = np.linalg.norm(nskel_jnts[link_seg[0]]-nskel_jnts[link_seg[1]])
        link_lengths.append(l1)

    np_link_lengths = np.array(link_lengths)
    return np_link_lengths

def get_delta_inc (pts_3d, length_array, vnect_start_idx, vnect_end_idx, avt_seg_idx):

    al = length_array[avt_seg_idx]
    start = pts_3d[vnect_start_idx]
    end = pts_3d[vnect_end_idx]
    vl = np.linalg.norm(start - end)
    direction = (end - start) / vl

    new_pos = start + direction * al
    delta_inc = new_pos - end

    return delta_inc

def adjust_link_lengths(pts_3d, length_array):

    new_pts_3d = np.zeros((pts_3d.shape[0], pts_3d.shape[1]))
    new_pts_3d[14] = pts_3d[14]
    # Hips
    delta_inc = get_delta_inc(pts_3d, length_array, 14, 15, 3)
    j_deltas = [15, 0, 1, 16, 2, 3, 4, 17, 5, 6, 7, 18]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Up Hips
    delta_inc = get_delta_inc(pts_3d, length_array, 15, 1, 2)
    j_deltas = [0, 1, 16, 2, 3, 4, 17, 5, 6, 7, 18]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Neck
    delta_inc = get_delta_inc(pts_3d, length_array, 1, 16, 1)
    j_deltas = [16, 0]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Head
    delta_inc = get_delta_inc(pts_3d, length_array, 16, 0, 0)
    j_deltas = [0]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Right shoulder
    delta_inc = get_delta_inc(pts_3d, length_array, 1, 2, 4)
    j_deltas = [2, 3, 4, 17]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Right upper arm
    delta_inc = get_delta_inc(pts_3d, length_array, 2, 3, 5)
    j_deltas = [3, 4, 17]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Right arm
    delta_inc = get_delta_inc(pts_3d, length_array, 3, 4, 6)
    j_deltas = [4, 17]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Right hand
    delta_inc = get_delta_inc(pts_3d, length_array, 4, 17, 7)
    j_deltas = [17]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Left shoulder
    delta_inc = get_delta_inc(pts_3d, length_array, 1, 5, 8)
    j_deltas = [5, 6, 7, 18]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Left upper arm
    delta_inc = get_delta_inc(pts_3d, length_array, 5, 6, 9)
    j_deltas = [6, 7, 18]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Left arm
    delta_inc = get_delta_inc(pts_3d, length_array, 6, 7, 10)
    j_deltas = [7, 18]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Left hand
    delta_inc = get_delta_inc(pts_3d, length_array, 7, 18, 11)
    j_deltas = [18]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Right hips
    delta_inc = get_delta_inc(pts_3d, length_array, 14, 8, 12)
    j_deltas = [8, 9, 10, 19]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Right up leg
    delta_inc = get_delta_inc(pts_3d, length_array, 8, 9, 13)
    j_deltas = [9, 10, 19]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Right leg
    delta_inc = get_delta_inc(pts_3d, length_array, 9, 10, 14)
    j_deltas = [10, 19]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Right foot
    delta_inc = get_delta_inc(pts_3d, length_array, 10, 19, 15)
    j_deltas = [19]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Left hips
    delta_inc = get_delta_inc(pts_3d, length_array, 14, 11, 16)
    j_deltas = [11, 12, 13, 20]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Left up leg
    delta_inc = get_delta_inc(pts_3d, length_array, 11, 12, 17)
    j_deltas = [12, 13, 20]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Left leg
    delta_inc = get_delta_inc(pts_3d, length_array, 12, 13, 18)
    j_deltas = [13, 20]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc
    # Left foot
    delta_inc = get_delta_inc(pts_3d, length_array, 13, 20, 19)
    j_deltas = [20]
    for j_delta in j_deltas:
        new_pts_3d[j_delta] = pts_3d[j_delta] + delta_inc

    return new_pts_3d

def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0]; # total points
    #print(A)
    #print(B)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.transpose(AA) @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        #print ("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = - R @ centroid_A.T + centroid_B.T
    #print (t)

    return R, t


# asynchronize plots
fig = plt.figure()

# fig2 = plt.figure()
# ax = Axes3D(fig2)
plt.ion()


INPUT_SIZE = 368

# get avatar rest pose bones
avatar_rest_pose_file = "/Users/jsanchez/Software/gitprojects/mocap_studio/avatar_rest_pose.npy"
avatar_rest_pose = np.load(avatar_rest_pose_file)

avatar_links_length = compute_avatar_links_length(avatar_rest_pose)

print(avatar_links_length)

# Initialize kalman filter

NUM_JOINTS = 21
list_KFs = []
for i in range(NUM_JOINTS):
    KF = kalman.KF2d( dt = 5 ) # time interval: '1 frame'
    init_P = 1*np.eye(4, dtype=np.float) # Error cov matrix
    init_x = np.array([0,0,0,0], dtype=np.float) # [x loc, x vel, y loc, y vel]
    dict_KF = {'KF':KF,'P':init_P,'x':init_x}
    list_KFs.append(dict_KF)

old_jnts_2d = np.zeros((21,2))
for f in range(num_files):
# for f in range(0,20):

    start = time.time()

    # read image file
    img_file = "%s/%04d.jpg" % (img_path, f)
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    # print(height)
    # print(width)
    # scale_x = width / INPUT_SIZE
    # scale_y = height / INPUT_SIZE
#    img_square = img_scale_squarify(img, INPUT_SIZE)

    # read joints 2d/3d
    jnts_2d_file = "%s/%04d.npy" % (jnts_2d_path, f)
    jnts_2d = np.load(jnts_2d_file)

    if f == 0 : # first frame dist = 0
        for i, jnt_2d in enumerate(jnts_2d):
            old_jnts_2d[i] = jnt_2d
    else:
        for i, jnt_2d in enumerate(jnts_2d):
            jnts_dist = np.linalg.norm(jnt_2d - old_jnts_2d[i])
            if jnts_dist > 25.0:
                jnts_2d[i] = old_jnts_2d[i]
            else:
                old_jnts_2d[i] = jnt_2d


    list_estimate = [] # kf filtered keypoints

    for i, jnt_2d in enumerate(jnts_2d):

        z = np.array( [jnt_2d[0], jnt_2d[1]], dtype=np.float)

        KF = list_KFs[i]['KF']
        x  = list_KFs[i]['x']
        P  = list_KFs[i]['P']
        
        x, P, filtered_point = KF.process(x, P, z)

        list_KFs[i]['KF'] = KF
        list_KFs[i]['x']  = x
        list_KFs[i]['P']  = P

        # # visibility
        # v = 0 if filtered_point[0] == 0 and filtered_point[1] == 0 else 2
        # list_estimate.extend(list(filtered_point) + [v]) # x,y,v
        list_estimate.extend(list(filtered_point)) # x,y,v

    np_estimate_2d =  np.array(list_estimate)
    estimate_jnts_2d = np.reshape(np_estimate_2d, (jnts_2d.shape[0], jnts_2d.shape[1]))

    jnts_3d_file = "%s/%04d.npy" % (jnts_3d_path, f)
    jnts_3d = np.load(jnts_3d_file)

    # find scale ratio
    length_leg_vnect = np.linalg.norm(jnts_3d[11]-jnts_3d[12])
    length_leg_avt = np.linalg.norm(avatar_rest_pose[19]-avatar_rest_pose[20])
    scale_f = length_leg_avt / length_leg_vnect

    jnts_3d = jnts_3d * scale_f

    # rotate joints from z-forware y-up (vnect) to z-up y-forward (blender)
    mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    jnts_3d = np.matmul(jnts_3d, mat)
    # pts_vnect = np.array([jnts_3d[8], jnts_3d[11], jnts_3d[1]])
    # mid_point = (avatar_rest_pose[8] +  avatar_rest_pose[12]) / 2
    # pts_avatar = np.array([avatar_rest_pose[16], avatar_rest_pose[19], mid_point])

    # init_R, initT = rigid_transform_3D (pts_vnect, pts_avatar)
    # jnts_3d = np.matmul(jnts_3d, init_R)
    # # jnts_3d = jnts_3d + initT

    hips_avatar = avatar_rest_pose[0]
    jnts_3d = jnts_3d + hips_avatar
    print(jnts_3d.shape)


    # adjust lengths to blender skeleton lengths
    jnts_3d = adjust_link_lengths (jnts_3d, avatar_links_length)

    # # check lengths are equal
    # print("Length check")
    # print(avatar_links_length[3])
    # print(np.linalg.norm(jnts_3d[15]-jnts_3d[14]))


    # ----- Plot section -----

    # draw_joints_2s(img, jnts_2d, 1.0, 1.0)
    draw_joints_2s(img, estimate_jnts_2d, 1.0, 1.0)

    # plt.figure(1)
    plt.axis('off')
    plt.imshow(img)

    # plt.show(block=False)

    # # --- plot 3d ---------

    # plt.figure(2)

    # X = jnts_3d[:,0]
    # Y = jnts_3d[:,1]
    # Z = jnts_3d[:,2]

    # ax.scatter(X, Y, Z, c='b', label='predictions')  # gt
    # # ax.set_xlabel('X-axis')
    # # ax.set_xlabel('Y-axis')
    # # ax.set_xlabel('Z-axis')
    # ax.legend()

    # # # normalize axis visualization
    # # X = jnts_3d[:,0]
    # # Y = jnts_3d[:,1]
    # # Z = jnts_3d[:,2]
    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    # mid_x = (X.max()+X.min()) * 0.5
    # mid_y = (Y.max()+Y.min()) * 0.5
    # mid_z = (Z.max()+Z.min()) * 0.5

    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # for l in range(0, len(vnect_links)):
    # # for l in range(0, 1):
    #     line_x = [X[vnect_links[l][0]], X[vnect_links[l][1]]]
    #     line_y = [Y[vnect_links[l][0]], Y[vnect_links[l][1]]]
    #     line_z = [Z[vnect_links[l][0]], Z[vnect_links[l][1]]]
    #     plt.plot( line_x, line_y, line_z, 'b')


    # AX = avatar_rest_pose[:,0]
    # AY = avatar_rest_pose[:,1]
    # AZ = avatar_rest_pose[:,2]

    # ax.scatter(AX, AY, AZ, c='g', label='predictions')  # gt
    # # ax.set_xlabel('X-axis')
    # # ax.set_xlabel('Y-axis')
    # # ax.set_xlabel('Z-axis')
    # ax.legend()


    # for l in range(0, len(avatar_links)):
    # # for l in range(0, 1):
    #     line_x = [AX[avatar_links[l][0]], AX[avatar_links[l][1]]]
    #     line_y = [AY[avatar_links[l][0]], AY[avatar_links[l][1]]]
    #     line_z = [AZ[avatar_links[l][0]], AZ[avatar_links[l][1]]]
    #     plt.plot( line_x, line_y, line_z, 'b')




    # # plt.axis('off')
    # # plt.imshow(img)

    # # plt.show()

    # ax.view_init(elev=9, azim=-68)
    #plt.show()
    #time.sleep(0.5)
    plt.show(block=False)
    plt.pause(0.01)
    # ax.cla()

    # plt.pause(0.01)
    # time.sleep(max(1./fps - (time.time() - start), 0))
    # plt.clf()






