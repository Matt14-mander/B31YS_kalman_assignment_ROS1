#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler, euler_from_quaternion

class SimpleKalmanFilterNode:
    def __init__(self):
        rospy.init_node('kalman_filter_simple', anonymous=True)

        # param
        self.dt = rospy.get_param('~dt', 0.1)  # Time step

        #subscriber
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        rospy.Subscriber('/fake_gps', Odometry, self.gps_callback)
        rospy.Subscriber('/odom1', Odometry, self.odom_callback)

        # Publisher 
        self.pub = rospy.Publisher('/kalman_estimate', Odometry, queue_size=10)

        # Initial State: [x, y, yaw]
        self.x = np.zeros((3,1))
        self.P = np.eye(3) * 0.1

        # Process and Measurement Noise Covariances
        q_pos = rospy.get_param('~q_pos', 1e-3)      
        q_yaw = rospy.get_param('~q_yaw', 1e-4)      
        r_gps = rospy.get_param('~r_gps', 1e-2)   
        self.r_odom_pos = rospy.get_param('~r_odom_pos', 1e-3)
        self.r_odom_yaw = rospy.get_param('~r_odom_yaw', 1e-4)   

        # Noise Covariances
        self.Q = np.diag([q_pos, q_pos, q_yaw]) # process noise. What should be the values here? 
        self.R_gps = np.diag([r_gps, r_gps]) # GPS measurement noise
        self.R_odom = np.diag([self.r_odom_pos, self.r_odom_pos, self.r_odom_yaw]) # odom1 measurement noise

        # Latest command velocities
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

        # Latest GPS measurement
        self.gps = None

        # Latest odom1 measurement
        self.odom1 = None

        # Timer for Kalman update
        rospy.Timer(rospy.Duration(self.dt), self.update_kalman)
        rospy.loginfo("Kalman Filter start")

    def cmd_vel_callback(self, msg):
        """Store the latest cmd velocities."""
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.yaw_rate = msg.angular.z

    def gps_callback(self, msg):
        """Store the latest GPS measurement."""
        self.gps = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y]
        ])

    def odom_callback(self, msg):
        """Odometry callback (not used in this simple filter)."""
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.odom1 = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
            [self.wrap_angle(yaw)]
        ])
        
    def wrap_angle(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi


    def update_kalman(self, event):
        """
        This is the main Kalman filter loop. In this function
        you should do a prediction plus a correction step. """
        dt = self.dt

        x = float(self.x[0,0])
        y = float(self.x[1,0])
        yaw = float(self.x[2,0])

        vx = float(self.vx)
        vy = float(self.vy)
        wz = float(self.yaw_rate)

        # --- Prediction ---
        dx = (vx*np.cos(yaw) - vy*np.sin(yaw)) * dt
        dy = (vx*np.sin(yaw) + vy*np.cos(yaw)) * dt
        dyaw = wz * dt

        x_pred = np.zeros((3,1))
        x_pred[0,0] = x + dx
        x_pred[1,0] = y + dy
        x_pred[2,0] = self.wrap_angle(yaw + dyaw)

        dxd_yaw = (-vx*np.sin(yaw) - vy*np.cos(yaw)) * dt
        dyd_yaw = ( vx*np.cos(yaw) - vy*np.sin(yaw)) * dt
        F = np.array([[1, 0, dxd_yaw],
                      [0, 1, dyd_yaw],
                      [0, 0, 1]])
        
        P_pred = F @ self.P @ F.T + self.Q
	
        self.x = x_pred
        self.P = P_pred

        # Correction-1
        if self.odom1 is not None:
            z_odom = self.odom1  # [x_odom, y_odom, yaw_odom]

            # z = H x + v
            H_odom = np.eye(3)

            # Innovation y = z - H x_pred
            y_innov_odom = z_odom - H_odom.dot(x_pred)
            y_innov_odom[2,0] = self.wrap_angle(y_innov_odom[2,0])

            # Innovation covariance S
            S_odom = H_odom.dot(P_pred).dot(H_odom.T) + self.R_odom

            # Kalman Gain K
            K_odom = P_pred.dot(H_odom.T).dot(np.linalg.inv(S_odom))

            # State update
            x_upd_odom = x_pred + K_odom.dot(y_innov_odom)
            x_upd_odom[2,0] = self.wrap_angle(x_upd_odom[2,0])

            # Covariance update
            I = np.eye(3)
            P_upd_odom = (I - K_odom.dot(H_odom)).dot(P_pred)

            self.x = x_upd_odom
            self.P = P_upd_odom

        # Correction-2
        if self.gps is not None:
            z = self.gps   # [x_gps, y_gps]

            # z = H x + v
            H = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ])

            # Innovation y = z - H x_pred
            y_innov = z - H.dot(x_pred)

            # Innovation covariance S
            S = H.dot(P_pred).dot(H.T) + self.R_gps

            # Kalman Gain K
            K = P_pred.dot(H.T).dot(np.linalg.inv(S))

            # State update
            x_upd = x_pred + K.dot(y_innov)
            x_upd[2,0] = self.wrap_angle(x_upd[2,0])

            # Covariance update
            I = np.eye(3)
            P_upd = (I - K.dot(H)).dot(P_pred)

            self.x = x_upd
            self.P = P_upd

        self.publish_estimate()

    def publish_estimate(self):
        """Publish the current state estimate as Odometry message."""
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "odom"

        msg.pose.pose.position.x = float(self.x[0])
        msg.pose.pose.position.y = float(self.x[1])

        q = quaternion_from_euler(0,0,float(self.x[2]))
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        msg.twist.twist.linear.x = float(self.vx)
        msg.twist.twist.linear.y = float(self.vy)
        msg.twist.twist.angular.z = float(self.yaw_rate)

        self.pub.publish(msg)


if __name__ == '__main__':
    try:
        node = SimpleKalmanFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

