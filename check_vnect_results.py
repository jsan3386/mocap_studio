import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math

vid_path = "/Users/jsanchez/Software/gitprojects/mocap_studio/vnect_res"
vid_name = "vid1"

img_path = "%s/%s/images" % (vid_path, vid_name)
jnts_2d_path = "%s/%s/joints_2d" % (vid_path, vid_name)
jnts_3d_path = "%s/%s/joints_3d" % (vid_path, vid_name)

# num files in folder
num_files = len([name for name in os.listdir(img_path)])
print(num_files)
fps = 20

# asynchronize plots
fig = plt.figure()

fig2 = plt.figure()
ax = Axes3D(fig2)
plt.ion()

# vnect_links
vnect_links = [[0, 16], [16, 1], [1, 2], [1, 5], [2, 3], [3, 4], [4, 17], [5, 6],
[6, 7], [7, 18], [1,15], [15, 14], [14, 8], [8, 9], [9, 10], [10, 19], [14, 11],
[11, 12], [12, 13], [13, 20]]


def img_scale(img, scale):
    """
    Resize a image by scale factor in both x and y directions.

    :param img: input image
    :param scale: scale factor, which is supposed to be side length of interpolated image / side length of source image
    :return: the scaled image
    """
    return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

def img_padding(img, box_size, color='black'):
    """
    Given the input image and side length of the box, put the image into the center of the box.

    :param img: the input color image, whose longer side is equal to box size
    :param box_size: the side length of the square box
    :param color: indicating the padding area color
    :return: the padded image
    """
    h, w = img.shape[:2]
    pad_num = 0
    if not color == 'black':
        if color == 'grey':
            pad_num = 128
    img_padded = np.ones((box_size, box_size, 3), dtype=np.uint8) * pad_num
    if h > w:
        img_padded[:, box_size // 2 - w // 2: box_size // 2 + int(np.ceil(w / 2)), :] = img
    else:  # h <= w
        img_padded[box_size // 2 - h // 2: box_size // 2 + int(np.ceil(h / 2)), :, :] = img
    return img_padded

def img_scale_squarify(img, box_size):
    """
    To scale and squarify the input image into a square box with fixed size.

    :param img: the input color image
    :param box_size: the length of the square box
    :return: the box image
    """
    h, w = img.shape[:2]
    scale = box_size / max(h, w)
    img_scaled = img_scale(img, scale)
    img_padded = img_padding(img_scaled, box_size)
    assert img_padded.shape == (box_size, box_size, 3), 'padded image shape invalid'
    return img_padded

def draw_joints_2s(test_img, joints, scale_x, scale_y):


    for joint in joints:
        # pt_x = int(joint[0] * scale_x)
        # pt_y = int(joint[1] * scale_y)
        pt_x = joint[0]
        pt_y = joint[1]

        cv2.circle(test_img, center=(pt_y, pt_x), radius=3, color=[0,0,255], thickness=-1)

    # Plot limb colors
    for link_idx, link in enumerate(vnect_links):

        print(link_idx)
        print(link)

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
            limb_color = map(lambda x: x + 35 * (link_idx % 4), [139, 53, 255])
            cv2.fillConvexPoly(test_img, polygon, color=limb_color)


INPUT_SIZE = 368


# for f in range(num_files):
for f in range(0,1):

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

    jnts_3d_file = "%s/%04d.npy" % (jnts_3d_path, f)
    jnts_3d = np.load(jnts_3d_file)

    print(jnts_2d.shape)
    print(jnts_3d.shape)

    draw_joints_2s(img, jnts_2d, 1.0, 1.0)

    plt.figure(1)
    plt.axis('off')
    plt.imshow(img)

    plt.show(block=False)

    plt.figure(2)

    X = jnts_3d[:,0]
    Y = jnts_3d[:,1]
    Z = jnts_3d[:,2]

    ax.scatter(X, Y, Z, c='b', label='predictions')  # gt
    # ax.set_xlabel('X-axis')
    # ax.set_xlabel('Y-axis')
    # ax.set_xlabel('Z-axis')
    ax.legend()



    # # normalize axis visualization
    # X = jnts_3d[:,0]
    # Y = jnts_3d[:,1]
    # Z = jnts_3d[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    for l in range(0, len(vnect_links)):
    # for l in range(0, 1):
        line_x = [X[vnect_links[l][0]], X[vnect_links[l][1]]]
        line_y = [Y[vnect_links[l][0]], Y[vnect_links[l][1]]]
        line_z = [Z[vnect_links[l][0]], Z[vnect_links[l][1]]]
        plt.plot( line_x, line_y, line_z, 'b')


    # plt.axis('off')
    # plt.imshow(img)

    # plt.show()

    ax.view_init(elev=9, azim=-68)
    #plt.show()
    #time.sleep(0.5)
    plt.show(block=True)
    plt.pause(0.01)
    ax.cla()

    # plt.pause(0.01)
    # time.sleep(max(1./fps - (time.time() - start), 0))
    # plt.clf()






