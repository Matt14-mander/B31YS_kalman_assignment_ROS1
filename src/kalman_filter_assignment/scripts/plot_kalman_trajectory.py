#!/usr/bin/env python3
import rospy
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry

# 记录数据
gps_x, gps_y = [], []
kalman_x, kalman_y = [], []
gt_x, gt_y = [], []

def gps_callback(msg):
    gps_x.append(msg.pose.pose.position.x)
    gps_y.append(msg.pose.pose.position.y)

def kalman_callback(msg):
    kalman_x.append(msg.pose.pose.position.x)
    kalman_y.append(msg.pose.pose.position.y)

def gt_callback(msg):
    gt_x.append(msg.pose.pose.position.x)
    gt_y.append(msg.pose.pose.position.y)

if __name__ == '__main__':
    rospy.init_node('plot_kalman_trajectory')

    # 订阅三个话题（根据实际话题名改）
    rospy.Subscriber('/fake_gps', Odometry, gps_callback)
    rospy.Subscriber('/kalman_estimate', Odometry, kalman_callback)
    rospy.Subscriber('/ground_truth', Odometry, gt_callback)

    print("Collecting data... Press Ctrl+C to stop.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    print("Plotting trajectories...")
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
    plt.show()
    print("Saved figure as kalman_trajectory.png")
