#!/usr/bin/env python3
import numpy
import random
import sys

import moveit_msgs.msg
import moveit_msgs.srv
import rclpy
from rclpy.node import Node
import rclpy.duration
import transforms3d._gohlketransforms as tf
import transforms3d
from urdf_parser_py.urdf import URDF
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform, PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import copy
import math
from rclpy.duration import Duration
from rclpy.clock import Clock
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def convert_to_message(T):
    t = Pose()
    position, Rot, _, _ = transforms3d.affines.decompose(T)
    orientation = transforms3d.quaternions.mat2quat(Rot)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[1]
    t.orientation.y = orientation[2]
    t.orientation.z = orientation[3]
    t.orientation.w = orientation[0]        
    return t

class MoveArm(Node):
    def __init__(self):
        super().__init__('move_arm')

        #Loads the robot model, which contains the robot's kinematics information
        self.ee_goal = None
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
        #Loads the robot model, which contains the robot's kinematics information
        self.declare_parameter(
            'rd_file', rclpy.Parameter.Type.STRING)
        robot_desription = self.get_parameter('rd_file').value
        with open(robot_desription, 'r') as file:
            robot_desription_text = file.read()
        self.robot = URDF.from_xml_string(robot_desription_text)
        self.base = self.robot.get_root()
        self.get_joint_info()
        self.marker_pub_s = self.create_publisher(Marker, '/visualization_marker_start', 10)
        self.marker_pub_g = self.create_publisher(Marker, '/visualization_marker_goal', 10)

        self.service_cb_group1 = MutuallyExclusiveCallbackGroup()
        self.service_cb_group2 = MutuallyExclusiveCallbackGroup()
        self.q_current = []

        # Wait for moveit IK service
        self.ik_service = self.create_client(moveit_msgs.srv.GetPositionIK, '/compute_ik', callback_group=self.service_cb_group1)
        while not self.ik_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for IK service...')
        self.get_logger().info('IK service ready')

        # Wait for validity check service
        self.state_valid_service = self.create_client(moveit_msgs.srv.GetStateValidity, '/check_state_validity',
                                                      callback_group=self.service_cb_group2)
        while not self.state_valid_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for state validity service...')
        self.get_logger().info('State validity service ready')

        # MoveIt parameter
        self.group_name = 'arm'
        self.get_logger().info(f'child map: \n{self.robot.child_map}')

        #Subscribe to topics
        self.sub_joint_states = self.create_subscription(JointState, '/joint_states', self.get_joint_state, 10)
        self.goal_cb_group = MutuallyExclusiveCallbackGroup()
        self.sub_goal = self.create_subscription(Transform, '/motion_planning_goal', self.motion_planning_cb, 2,
                                                 callback_group=self.goal_cb_group)
        self.current_obstacle = "NONE"
        self.sub_obs = self.create_subscription(String, 'obstacles', self.get_obstacle, 10)

        #Set up publisher
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)
        self.joint_trajectory = JointTrajectory()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(0.1, self.motion_planning_timer, callback_group=self.timer_cb_group)
        self.smooth_path = []
    
    def get_joint_state(self, msg):
        '''This callback provides you with the current joint positions of the robot 
        in member variable q_current.
        '''
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])

    def get_obstacle(self, msg):
        '''This callback provides you with the name of the current obstacle which
        exists in the RVIZ environment. Options are "None", "Simple", "Hard",
        or "Super". '''
        self.current_obstacle = msg.data

    def motion_planning_cb(self, ee_goal):
        self.get_logger().info("Motion planner goal received.")
        if self.ee_goal is not None:
            self.get_logger().info("Motion planner busy. Please try again later.")
            return
        self.ee_goal = ee_goal

    def motion_planning_timer(self):
        if self.ee_goal is not None:
            self.get_logger().info("Calling motion planner")            
            self.motion_planning(self.ee_goal)
            self.ee_goal = None
            self.get_logger().info("Motion planner done")

    def generate_random_config(self):
        q_0 = [0] * len(self.joint_names)
        for i in range(0,len(self.joint_names)):
            q_0[i] = random.uniform(-math.pi,math.pi)
        return numpy.array(q_0).astype(float)
    
    def expand_tree(self, tree, q_rand, step_size):
        if numpy.linalg.norm(numpy.array(tree.q) - numpy.array(q_rand)) < step_size:
            return RRTBranch(tree, q_rand)
        else:
            # Compute the direction from tree.q to q_rand
            direction = numpy.array(q_rand) - numpy.array(tree.q)
            direction = direction / numpy.linalg.norm(direction)
            q_new = numpy.array(tree.q) + step_size * direction
                # Create a new RRTBranch instance with tree as the parent and q_new as the configuration
            new_node = RRTBranch(tree, q_new.tolist())
        
        return new_node
    
    def find_nearby_nodes(self, tree, r):
        r_array = numpy.array(r)
        shortest_distance = float('inf')  # Initialize to infinity
        closest_point = None

        # Iterate through the tree to find the closest point to r
        for node in tree:
            node_q_array = numpy.array(node.q)
            distance = numpy.linalg.norm(r_array - node_q_array)
            if distance < shortest_distance:
                shortest_distance = distance
                closest_point = node

        return closest_point

    def message_to_transform(self,transform_msg):
        position = transform_msg.translation
        orientation = transform_msg.rotation
        T = transforms3d.affines.compose([position.x, position.y, position.z],
                                        transforms3d.quaternions.quat2mat([orientation.w, orientation.x, orientation.y, orientation.z]),
                                        [1, 1, 1])
        return T
    
    def extend_tree(self, tree, q_rand, step_size):
        closest_point = self.find_nearby_nodes(tree, q_rand)
        new_node = self.expand_tree(closest_point, q_rand, step_size)
        if self.is_state_valid(new_node.q):
            tree.append(new_node)
            return new_node
        else:
            return None
        
    def shortcut(self, path):
        new_path = path
        max_iterations = len(path) // 2
        for i in range(0, max_iterations):
            q1 = random.randint(0, len(path) - 1)
            q2 = random.randint(0, len(path) - 1)
            if self.is_segment_valid(path[q1], path[q2]):
                self.smooth_path.reverse()
                new_path = path[:q1 + 1] + self.smooth_path + path[q2:]
                return new_path
        else:
            return path
    
    def is_segment_valid(self, q1, q2):
        # Compute the direction from q1 to q2
        if (q1 == q2).all():
            return True
        direction = numpy.array(q2) - numpy.array(q1)
        distance = numpy.linalg.norm(direction)
        direction = direction / distance
        self.smooth_path = []
        # Check if the segment is valid by sampling points along the segment
        for i in range(0, int(distance / 0.1)):
            q = numpy.array(q1) + 0.1 * i * direction
            if not self.is_state_valid(q):
                return False
            else:
                self.smooth_path.append(q)
        return True
        
    def construct_path(self, goal_q, start_node, goal_node):
        path = []
        branch = start_node
        while branch is not None:
            path.append(branch.q)
            branch = branch.parent
        path = path[::-1]
        branch = goal_node.parent
        while branch is not None:
            path.append(branch.q)
            branch = branch.parent

        if (path[-1] != goal_q).any():  # check if the last point is not the goal
            path = path[::-1]

        #path = self.shortcut(path)
        
        return path
    
    def RRTConnect(self, q_goal, q_start):
        q_start = self.q_current
        q = q_goal
        start_tree = [RRTBranch(None, q_start)]
        goal_tree = [RRTBranch(None, q_goal)]
        while True:
            if self.current_obstacle == "NONE":
                max_iterations = 10000000
                step_size = 0.1
                bias = 10
            elif self.current_obstacle == "SIMPLE":
                max_iterations = 10000000
                step_size = 0.1
                bias = 10
            elif self.current_obstacle == "HARD":
                max_iterations = 100000000
                step_size = 0.1
                bias = 5
            elif self.current_obstacle == "SUPER":
                max_iterations = 100000000
                step_size = 0.1
                bias = 5
            #self.get_logger().info(self.current_obstacle)
            for i in range(max_iterations):
                if i % bias == 0:
                    q_rand = q_goal
                else: 
                    q_rand = self.generate_random_config()
                
                # Extend the start tree towards q_rand
                new_start_node = self.extend_tree(start_tree, q_rand, step_size)

                
                if new_start_node is not None:
                    # Try to connect the goal tree to the new node in the start tree
                    new_goal_node = self.extend_tree(goal_tree, new_start_node.q, step_size)
                    if i % 2 == 0:
                        self.visualize_gtree(goal_tree)
                        self.visualize_stree(start_tree)
                    else:
                        self.visualize_gtree(start_tree)
                        self.visualize_stree(goal_tree)
                    if new_goal_node is not None and numpy.linalg.norm(numpy.array(new_start_node.q) - numpy.array(new_goal_node.q)) < 0.1:
                        # If the trees are connected, construct the path
                        
                        return self.construct_path(q, new_start_node, new_goal_node)
                
                # Swap the roles of the start tree and the goal tree
                start_tree, goal_tree = goal_tree, start_tree
                q_goal, q_start = q_start, q_goal
    
   
    def motion_planning(self, ee_goal: Transform): 
        '''Callback function for /motion_planning_goal. This is where you will
        implement your RRT motion planning which is to generate a joint
        trajectory for your manipulator. You are welcome to add other functions
        to this class (i.e. an is_segment_valid" function will likely come in 
        handy multiple times in the motion planning process and it will be 
        easiest to make this a seperate function and then call it from motion
        planning). You may also create trajectory shortcut and trajectory 
        sample functions if you wish, which will also be called from the 
        motion planning function.

        Args: 
            ee_goal: Transform() object describing the desired base to 
            end-effector transformation 
        '''
        q_start = self.q_current
        #self.get_logger().info("start with initial position")
        #self.get_logger().info(str(q_start))
        q_goal = self.IK(self.message_to_transform(ee_goal))
        #self.get_logger().info("goal position")
        #self.get_logger().info(str(q_goal))
        path = self.RRTConnect(q_goal, q_start)

        if path is None:
            self.get_logger().info("RRT fail")
        else:
            self.get_logger().info("RRT success")
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_names 
            for q in path:
                #self.get_logger().info(str(q))  # debug
                point = JointTrajectoryPoint()
                point.positions = q.tolist()
                point.time_from_start = rclpy.duration.Duration(seconds=1.0).to_msg() # set the time for each point
                trajectory_msg.points.append(point)
            self.pub.publish(trajectory_msg)

    def IK(self, T_goal):
        """ This function will perform IK for a given transform T of the 
        end-effector. It .

        Returns:
            q: returns a list q[] of values, which are the result 
            positions for the joints of the robot arm, ordered from proximal 
            to distal. If no IK solution is found, it returns an empy list
        """

        req = moveit_msgs.srv.GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.robot_state = moveit_msgs.msg.RobotState()
        req.ik_request.robot_state.joint_state.name = self.joint_names
        req.ik_request.robot_state.joint_state.position = list(numpy.zeros(self.num_joints))
        req.ik_request.robot_state.joint_state.velocity = list(numpy.zeros(self.num_joints))
        req.ik_request.robot_state.joint_state.effort = list(numpy.zeros(self.num_joints))
        req.ik_request.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped = PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = 'base'
        req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
        req.ik_request.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()
        
        self.get_logger().info('Sending IK request...')
        res = self.ik_service.call(req)
        self.get_logger().info('IK request returned')
        
        q = []
        if res.error_code.val == res.error_code.SUCCESS:
            q = res.solution.joint_state.position
        for i in range(0,len(q)):
            while (q[i] < -math.pi): q[i] = q[i] + 2 * math.pi
            while (q[i] > math.pi): q[i] = q[i] - 2 * math.pi
        return numpy.array(q).astype(float)

    
    def get_joint_info(self):
        '''This is a function which will collect information about the robot which
        has been loaded from the parameter server. It will populate the variables
        self.num_joints (the number of joints), self.joint_names and
        self.joint_axes (the axes around which the joints rotate)
        '''
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]
            if current_joint.type != 'fixed':
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link
        self.get_logger().info('Num joints: %d' % (self.num_joints))


    
    def is_state_valid(self, q):
        """ This function checks if a set of joint angles q[] creates a valid state,
        or one that is free of collisions. The values in q[] are assumed to be values
        for the joints of the UR5 arm, ordered from proximal to distal.

        Returns:
            bool: true if state is valid, false otherwise
        """
        req = moveit_msgs.srv.GetStateValidity.Request()
        req.group_name = self.group_name
        req.robot_state = moveit_msgs.msg.RobotState()
        req.robot_state.joint_state.name = self.joint_names
        req.robot_state.joint_state.position = list(q)
        req.robot_state.joint_state.velocity = list(numpy.zeros(self.num_joints))
        req.robot_state.joint_state.effort = list(numpy.zeros(self.num_joints))
        req.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()

        res = self.state_valid_service.call(req)

        return res.valid
    

    def forward_kinematics(self, joint_values):
            joint_transforms = []

            link = self.robot.get_root()
            T = tf.identity_matrix()

            while True:
                if link not in self.robot.child_map:
                    break

                (joint_name, next_link) = self.robot.child_map[link][0]
                joint = self.robot.joint_map[joint_name]

                T_l = numpy.dot(tf.translation_matrix(joint.origin.xyz), tf.euler_matrix(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2], 'rxyz'))
                T = numpy.dot(T, T_l)

                if joint.type != "fixed":
                    joint_transforms.append(T)
                    q_index = self.joint_names.index(joint_name)
                    T_j = tf.rotation_matrix(joint_values[q_index], numpy.asarray(joint.axis))
                    T = numpy.dot(T, T_j)

                link = next_link
            return joint_transforms, T #where T = b_T_ee
    def visualize_stree(self, tree):
            marker = Marker()
            marker.header.frame_id = "base_link"  
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rrt_tree"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.01  
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0

            for branch in tree:
                if branch.parent is not None:
                    _, T1 = self.forward_kinematics(branch.parent.q)
                    start_position = tf.translation_from_matrix(T1)
                    _, T2 = self.forward_kinematics(branch.q)
                    end_position = tf.translation_from_matrix(T2)
                
                    start = Point()
                    start.x, start.y, start.z = start_position
                
                    end = Point()
                    end.x, end.y, end.z = end_position
                
                    marker.points.append(start)
                    marker.points.append(end)

            self.marker_pub_s.publish(marker)


    def visualize_gtree(self, tree):
            marker = Marker()
            marker.header.frame_id = "base_link"  
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rrt_tree"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.01  
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            for branch in tree:
                if branch.parent is not None:
                    _, T1 = self.forward_kinematics(branch.parent.q)
                    start_position = tf.translation_from_matrix(T1)
                    _, T2 = self.forward_kinematics(branch.q)
                    end_position = tf.translation_from_matrix(T2)
                
                    start = Point()
                    start.x, start.y, start.z = start_position
                
                    end = Point()
                    end.x, end.y, end.z = end_position
                
                    marker.points.append(start)
                    marker.points.append(end)

            self.marker_pub_g.publish(marker)

class RRTBranch(object):
    '''This is a class which you can use to keep track of your tree branches.
    It is easiest to do this by appending instances of this class to a list 
    (your 'tree'). The class has a parent field and a joint position field (q). 
    
    You can initialize a new branch like this:
        RRTBranch(parent, q)
    Feel free to keep track of your branches in whatever way you want - this
    is just one of many options available to you.
    '''
    def __init__(self, parent, q):
        self.parent = parent
        self.q = numpy.array(q).astype(float)

def main(args=None):
    rclpy.init(args=args)
    ma = MoveArm()
    ma.get_logger().info("Move arm initialization done")
    executor = MultiThreadedExecutor()
    executor.add_node(ma)
    executor.spin()
    ma.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        

