#coding=utf-8

bl_info = {
    'name': "MocapStudio",
    'author': "Jordi Sanchez-Riera",
    'version': (0, 1, 0),
    "blender": (2, 80, 0),
    'location': "View3D",
    'description': "Mocap studio for the avatar suite",
    'warning': '',
    'wiki_url': '',
    'category': 'Mocap'
}

import os
import sys
import bpy

import math
from mathutils import Vector, Quaternion, Matrix

import zmq


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    #buf = memoryview(msg)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


class AVATAR_OT_StreamingPose(bpy.types.Operator):
    bl_idname = "avt.streaming_pose"
    bl_label = "Connect socket"  # Display name in the interface.
#    bl_options = {'REGISTER', 'UNDO'} 
    bl_options = {'REGISTER'} 

    def execute(self, context):  # execute() is called when running the operator.
        global mAvt

        if not context.window_manager.socket_connected:
            self.zmq_ctx = zmq.Context().instance()  # zmq.Context().instance()  # Context
            bpy.types.WindowManager.socket = self.zmq_ctx.socket(zmq.SUB)
            bpy.types.WindowManager.socket.connect(f"tcp://127.0.0.1:5667")  # publisher connects to this (subscriber)
            bpy.types.WindowManager.socket.setsockopt(zmq.SUBSCRIBE, ''.encode('ascii'))
            print("Waiting for data...")

            # poller socket for checking server replies (synchronous)
            self.poller = zmq.Poller()
            self.poller.register(bpy.types.WindowManager.socket, zmq.POLLIN)

            # let Blender know our socket is connected
            context.window_manager.socket_connected = True
            mAvt.frame = 1
            mAvt.start_origin = context.window_manager.start_origin
            mAvt.write_timeline = context.window_manager.write_timeline

            bpy.app.timers.register(self.timed_msg_poller)

        # stop ZMQ poller timer and disconnect ZMQ socket
        else:
            # cancel timer function with poller if active
            if bpy.app.timers.is_registered(self.timed_msg_poller):
                bpy.app.timers.unregister(self.timed_msg_poller())

            try:
                # close connection
                bpy.types.WindowManager.socket.close()
                print("Subscriber socket closed")
                # remove reference
            except AttributeError:
                print("Subscriber socket was not active")

            # let Blender know our socket is disconnected
            bpy.types.WindowManager.socket = None
            context.window_manager.socket_connected = False
            context.window_manager.pid = 0

        return {'FINISHED'}  # Lets Blender know the operator finished successfully.

    def timed_msg_poller(self):  # context
        global mAvt
        socket_sub = bpy.types.WindowManager.socket
#        write_timeline = bpy.types.WindowManager.write_timeline
#        start_origin = bpy.types.WindowManager.start_origin
        # only keep running if socket reference exist (not None)
        if socket_sub:
            # get sockets with messages (0: don't wait for msgs)
            sockets = dict(self.poller.poll(0))
            # check if our sub socket has a message
            if socket_sub in sockets:
                # get the message
                points3d = recv_array(socket_sub)
                #print(points3d)
                # When using points obtained from Matlab
                # M_mb = motion_utils.get_trans_mat_blend_to_matlab()
                # pts_skel = np.matmul(points3d, M_mb)
                pts_skel = points3d
                if mAvt.start_origin:
                    # translate points
                    new_pts_skel = []
                    if mAvt.frame == 1:
                        hips = pts_skel[14,:]
                        mAvt.trans = hips - np.array(mAvt.hips_pos)
                    for pt in pts_skel:
                        new_pts_skel.append( [pt[0]-mAvt.trans[0], pt[1]-mAvt.trans[1], pt[2]-mAvt.trans[2]])
                    pts_skel = np.array(new_pts_skel)

                # set skeleton rest position: MAYBE MOVE ALL THIS TO SERVER.PY IN ORDER TO MAKE FASTER UPDATES
                # slow version
                # motion_utils.set_rest_pose(mAvt.skel, mAvt.skel_ref, mAvt.list_bones)
                # motion_utils.calculate_rotations(mAvt.skel, pts_skel)
                # faster version
                motion_utils.set_rest_pose3(mAvt.skel, mAvt.list_matrices_basis, mAvt.list_matrices_local)
                motion_utils.calculate_rotations_fast(mAvt.skel, mAvt.list_nodes, pts_skel)


                if mAvt.write_timeline:
                    bpy.context.view_layer.update()
                    mAvt.skel.keyframe_insert(data_path = "location", index = -1, frame = mAvt.frame)
            
                    for bone in mAvt.list_bones:
                        mAvt.skel.pose.bones[bone].keyframe_insert(data_path = "rotation_quaternion", index = -1, frame = mAvt.frame)

                    mAvt.frame += 1

        # keep running
        return 0.001

class MOCAP_OT_StreamingPublisher(bpy.types.Operator):
    bl_idname = "avt.streaming_publisher"
    bl_label = "Start streaming"  # Display name in the interface.
#    bl_options = {'REGISTER', 'UNDO'} 
    bl_options = {'REGISTER'} 

    bpy.types.WindowManager.pid = IntProperty(default=0)

    def execute(self, context):  # execute() is called when running the operator.
        global avt_path

        if not context.window_manager.streaming:
            str_fps = str(context.window_manager.fps)
            # path_frames = "%s/motion/frames" % avt_path
            # path_frames = "/mnt/data/jsanchez/Blender/Renders/sequence_animation/goalie_throw/seq"
            path_frames = "/mnt/data/jordi_tf/Projects/Avatar/Results-Avatar"
            # path_frames = "/mnt/data/jordi_tf/Projects/Avatar/Results-Avatar/real_seq"
            prog = "%s/motion/server.py" % avt_path
            proc = subprocess.Popen(["python", prog, "-frames_mixamo", path_frames, str_fps]) 
            context.window_manager.pid = proc.pid
            context.window_manager.streaming = True
            mAvt.start_origin = context.window_manager.start_origin
            mAvt.write_timeline = context.window_manager.write_timeline

        else:
            if context.window_manager.pid != 0:
                os.kill(context.window_manager.pid, signal.SIGTERM)
            context.window_manager.streaming = False

        return {'FINISHED'}

class MOCAP_PT_MotionPanel(bpy.types.Panel):
    
    bl_idname = "MOCAP_PT_MotionPanel"
    bl_label = "Motion"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mocap"

    # NOTE:
    # For the moment we don't implement write_bvh. This can be done registering motin to timeline and the export to bvh

    bpy.types.WindowManager.socket_connected = BoolProperty(name="Connect status", description="Boolean", default=False)
    bpy.types.WindowManager.streaming = BoolProperty(name="Streaming status", description="Boolean", default=False)
    bpy.types.WindowManager.fps = IntProperty(name="FPS", description="Streaming frame rate", default=30, min=1, max=60)
#    bpy.types.WindowManager.write_bvh = BoolProperty(name = "wBvh", description="Start at origin", default = False)
    bpy.types.WindowManager.write_timeline = BoolProperty(name = "wTimeline", description="Start at origin", default = False)



    def draw(self, context):
        layout = self.layout
        obj = context.object
        wm = context.window_manager
        
        #row = layout.row()

#        row = layout.row()
#        layout.prop(wm, "write_bvh", text="Write BVH file")
        #layout.prop(context.scene, 'test_enum', text='enum property', icon='NLA')

        if not wm.socket_connected:
            layout.operator("avt.streaming_pose")  # , text="Connect Socket"
        else:
            layout.operator("avt.streaming_pose", text="Disconnect Socket")

        if not wm.socket_connected:
            return

        # row = layout.row()
        # layout.operator("avt.streaming_publisher")  # , text="Connect Socket"

#        row = layout.row()
        if not wm.streaming:
            layout.operator("avt.streaming_publisher")  # , text="Start streaming"
        else:
            layout.operator("avt.streaming_publisher", text="Stop streaming")

        layout.prop(wm, "fps")
        layout.prop(wm, "write_timeline", text="Write timeline keypoints")
        layout.prop(wm, "start_origin", text="Start at origin")

# def set_source_skeleton(self, context):
#     skel = self.skel_type


classes  = (
            MOCAP_PT_MotionPanel,
            MOCAP_OT_StreamingPose,
            MOCAP_OT_StreamingPublisher,
)

def register():

    from bpy.utils import register_class  
    for clas in classes:
        register_class(clas)


def unregister():
    from bpy.utils import unregister_class  
    for clas in classes:
        unregister_class(clas)


if __name__ == '__main__':
    register()
