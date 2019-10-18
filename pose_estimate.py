#!/usr/bin/env python
# pose of my body frame(x front, y left, z up) (NOT THE CAMERA FRAME(x left, y down, z front)) 
# relative to my inertial frame(defined at the centre point of the window)
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Pose
from enae788m_p3.msg import window_features
from scipy.spatial.transform import Rotation as R

pub = rospy.Publisher('pnp', Pose)
pose = Pose()

def pnp_callback(data):

    points_2d = np.float32([[data.centre[0], data.centre[1]], [data.corner1[0], data.corner1[1]],
      [data.corner2[0], data.corner2[1]], [data.corner3[0], data.corner3[1]],
      [data.corner4[0], data.corner4[1]]])  

    points_3d = np.float32([[0.0, 0.0, 0.0], [-390, -215, 0.0], [390, -215, 0.0], [420, 215, 0.0], [-420, 215, 0.0]]) 

    intrinsics = np.array([345.1095082193839, 344.513136922481, 315.6223488316934, 238.99403696680216]) #mm
    dist_coeff = np.array([-0.3232637683425793, 0.045757813401817116, 0.0024085161807053074, 0.003826902574202108])

    K = np.float64([ [intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]])

    # rvec and tvec will give you the position of the world frame(defined at the center of the window) relative to the camera frame 
    _res, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K, dist_coeff, None, None, False, cv2.SOLVEPNP_ITERATIVE)

    rmat = cv2.Rodrigues(rvec)
    print(len(rmat))

    print("rvec + {}".format(np.degrees(rvec)))
    print("rmat + {}".format(rmat))
    print("tvec + {}".format(tvec))

    cRb = np.transpose(np.array([0, -1, 0], [0, 0, -1], [1, 0, 0]))
    cRi = rmat
    bRi = np.transpose(cRb)*rmat
    bti = cRb*tvec

    iRb = np.transpose(bRi)
    itb = -np.transpose(bRi)*bti

    rmat_fin = cv2.Rodrigues(iRb)
    tvec_fin = itb

    pose.position.x = tvec_fin[0]
    pose.position.y = tvec_fin[1]
    pose.position.z = tvec_fin[2]

    rquat = R.from_rotvec(rvec_fin)
    rquat = rquat.as_quat()
    print(rquat)

    pose.orientation.w = rquat[3]
    pose.orientation.x = rquat[0]
    pose.orientation.y = rquat[1]
    pose.orientation.z = rquat[2]

    pub.publish(pose)
    rospy.loginfo(pose)

def pose_estimate():
    rospy.init_node('pose_estimate', anonymous=True)
    rospy.Subscriber('features2d', window_features, pnp_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        pose_estimate()
    except rospy.ROSInterruptException:
        pass