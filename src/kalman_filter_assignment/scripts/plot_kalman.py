#!/usr/bin/env python3
import rospy
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry

# Data storage
gps_x, gps_y, gps_t = [], [], []
kalman_x, kalman_y, kalman_t = [], [], []
gt_x, gt_y, gt_t = [], [], []

t0 = None

def get_time(stamp):
    global t0
    t = stamp.to_sec()
    if t0 is None:
        t0 = t
    return t - t0

def gps_callback(msg):
    gps_t.append(get_time(msg.header.stamp))
    gps_x.append(msg.pose.pose.position.x)
    gps_y.append(msg.pose.pose.position.y)

def kalman_callback(msg):
    kalman_t.append(get_time(msg.header.stamp))
    kalman_x.append(msg.pose.pose.position.x)
    kalman_y.append(msg.pose.pose.position.y)

def gt_callback(msg):
    gt_t.append(get_time(msg.header.stamp))
    gt_x.append(msg.pose.pose.position.x)
    gt_y.append(msg.pose.pose.position.y)

# main function
if __name__ == '__main__':
    rospy.init_node('plot_kalman_trajectory')

    # Subscribers
    rospy.Subscriber('/fake_gps', Odometry, gps_callback)
    rospy.Subscriber('/kalman_estimate', Odometry, kalman_callback)
    rospy.Subscriber('/odom', Odometry, gt_callback)

    # Process ROS callbacks
    print("Collecting data... Press Ctrl+C to stop.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    print("Plotting results...")

    # Figure 1：Trajectory X vs Y
    plt.figure(figsize=(8,6))
    plt.title("Kalman Filter vs GPS vs Ground Truth Trajectories")
    plt.xlabel("X position [m]")
    plt.ylabel("Y position [m]")
    plt.grid(True)

    if gt_x:
        plt.plot(gt_x, gt_y, 'r-', label='Ground Truth')
    if gps_x:
        plt.scatter(gps_x, gps_y, s=10, c='g', marker='o', label='GPS (Fake)')
    if kalman_x:
        plt.plot(kalman_x, kalman_y, 'b-', linewidth=2, label='Kalman Estimate')

    plt.legend()
    plt.tight_layout()
    plt.savefig("kalman_trajectory.png", dpi=200)
    print("Saved trajectory plot as kalman_trajectory.png")

    # Figure 2：Time Sequence plot
    plt.figure(figsize=(8,6))
    plt.title("Kalman Filter vs GPS vs Ground Truth (X Position over Time)")
    plt.xlabel("Time [s]")
    plt.ylabel("X position [m]")
    plt.grid(True)

    if gt_t:
        plt.plot(gt_t, gt_x, 'r-', label='Ground Truth')
    if gps_t:
        plt.plot(gps_t, gps_x, 'go', label='GPS (Fake)', markersize=3)
    if kalman_t:
        plt.plot(kalman_t, kalman_x, 'b-', linewidth=2, label='Kalman Estimate')

    plt.legend()
    plt.tight_layout()
    plt.savefig("kalman_time_plot.png", dpi=200)
    print("Saved time plot as kalman_time_plot.png")

    plt.show()
