#!/usr/bin/env python

import rospy
import time
import numpy as np
#import tf  # use tf to easily calculate orientations


from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose


global_INERTIALpos= Pose()

global_firstPnP=False



def quat_mult(a,b):
    
    c = np.array([0.,0.,0.,0.])
    c[0] = (a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3] )
    c[1] = (a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2] )
    c[2] = (a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1] )
    c[3] = (a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0] )
    return c

def callback_PnP(msg):
	global global_INERTIALpos
	global global_firstPnP
	rospy.loginfo(msg)
	global_INERTIALpos=msg
	global_firstPnP=True

def quat_invert(quaternion):
	quat_inv=quaternion
	quat_inv[1:]= -1*quaternion[1:]
	return quat_inv

def rotatebyQuat(vector,quaternion):
	
	v=np.array([0, vector[0] ,vector[1], vector[2]])
	quat_inv= quat_invert(quaternion)
	temp= quat_mult(quat_inv,v)
	vout= quat_mult(temp,quaternion)
	return vout[1:]

def FindVector():
	global global_INERTIALpos
	global global_firstPnP

	rospy.init_node('desired_vector_finder', anonymous=True, log_level=rospy.WARN)
	rospy.Subscriber('pnp',Pose, callback_PnP)
	pub_vect= rospy.Publisher('bebop/desired_vector',Twist,queue_size=1)

	#quat X to Y is the orientation of Y relative to X, so it is the oration to get a vector in X rotated into Y
	if global_firstPnP==True:
		quat_I_to_B= np.array([global_INERTIALpos.orientation.w, global_INERTIALpos.orientation.x, global_INERTIALpos.orientation.y, global_INERTIALpos.orientation.z])
		v_B_inI= np.array([global_INERTIALpos.position.x,global_INERTIALpos.position.y,global_INERTIALpos.position.z])
		
		v_des_inI= [0, 0, 1.5]

		#desired point wrt window in B
		v_des_inB= rotatebyQuat(v_des_inI, quat_I_to_B)
		#window wrt B in B
		v_I_inB= rotatebyQuat( -1*v_B_inI, quat_I_to_B)

		#yaw angle to look at window is the angle between x and the vector to the window
		yaw= np.atan2(V_I_inB[1],V_I_inB[0])
		#desired point wrt B in B
		v_desB_inB= v_I_inB + v_des_inB

		outTwist=Twist()
		
		outTwist.linear.x=v_desB_inB[0]
		outTwist.linear.y=v_desB_inB[1]
		outTwist.linear.z=v_desB_inB[2]
		outTwist.angular.x=0
		outTwist.angular.y=0
		outTwist.angular.z=yaw

		pub_vect.publish(outTwist)


	spin()


if __name__ == '__main__':
	FindVector()