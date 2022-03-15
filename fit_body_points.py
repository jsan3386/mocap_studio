import bpy
import scipy
from mathutils import Matrix, Vector
import numpy as np

# fit model points
# src: makehuman model
# trg: points from smpl

# 2 versions: 1 using FK, 2 using IK

def get_bone_head_position(obj, bone_name):
    return (obj.matrix_world @ Matrix.Translation(obj.pose.bones[bone_name].head)).to_translation()

def get_rotations (obj, bones_list):
    rots = []
    for bone in bones_list:
        pb = obj.pose.bones[bone]
        rot_mat = pb.matrix_basis.copy()
        eul_ang = rot_mat.to_euler()
        rots.append([eul_ang[0], eul_ang[1], eul_ang[2]])
    return np.array(rots)

def apply_rotations (obj, bones_list, rotations):
    for b_idx, bone in enumerate(bones_list):
        pb = obj.pose.bones[bone]
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = rotations[b_idx]

def get_skel_positions (obj, list_bones):
    list_pts = []
    if obj.name == "Avatar":
        b_idx = 0
    if obj.name == "Armature":
        b_idx = 1
    for bone_pair in list_bones:
        bone = bone_pair[b_idx]
        pt = get_bone_head_position(obj, bone)
        list_pts.append(pt)
    return np.array(list_pts)



def f (obj, list_bones_rot, list_bones_pos, rotations):
    # apply rotations
    apply_rotations(obj, list_bones_rot, rotations)
    return get_skel_positions(obj, list_bones_pos)


# 0 Makehuman; 1 SMPL
list_bones_pos    = [["Hips", "f_avg_Pelvis"],
                     ["RightUpLeg", "f_avg_R_Hip"],
                     ["LeftUpLeg", "f_avg_L_Hip"],
                     ["RightLeg", "f_avg_R_Knee"],
                     ["LeftLeg", "f_avg_L_Knee"],
                     ["RightFoot", "f_avg_R_Ankle"],
                     ["LeftFoot", "f_avg_L_Ankle"],
                     ["RightToeBase", "f_avg_R_Foot"],
                     ["LeftToeBase", "f_avg_L_Foot"],
                     ["LowerBack", "f_avg_Spine1"],
                     ["Spine", "f_avg_Spine2"],
                     ["RightArm", "f_avg_R_Shoulder"],
                     ["LeftArm", "f_avg_L_Shoulder"],
                     ["RightForeArm", "f_avg_R_Elbow"],
                     ["LeftForeArm", "f_avg_L_Elbow"],
                     ["RightHand", "f_avg_R_Wrist"],
                     ["LeftHand", "f_avg_L_Wrist"],
                     ["Neck", "f_avg_Head"]
                    ]


# bones control makehuman
list_bones_rot  = ["Hips",
                   "RightUpLeg",
                   "LeftUpLeg",
                   "RightLeg",
                   "LeftLeg",
                   "RightFoot",
                   "LeftFoot",
                   "LowerBack",
                   "Spine",
                   "RightArm",
                   "LeftArm",
                   "RightForeArm",
                   "LeftForeArm",
                   "Neck"
                  ]


# assume SMPL and Makehuman already in scene
mh = bpy.data.objects["Avatar"]
smpl = bpy.data.objects["Armature"]

# get smpl and makehuman 3d points
mh_pts = get_skel_positions(mh, list_bones_pos)
smpl_pts = get_skel_positions(smpl, list_bones_pos)


# fit points
# rotation angles
ini_rot_ang = get_rotations(mh, list_bones_rot)
print(ini_rot_ang)
ini_rot_ang[10,:] = [0,0,0]

apply_rotations(mh, list_bones_rot, ini_rot_ang)



