from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2wRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        dof_idx_leg = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14] # FL -> FR -> RL -> RR
        dof_idx_wheel = [3, 7, 11, 15]
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]            

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'FL_foot_joint': 0.,   # [rad]
            'RL_foot_joint': 0.,   # [rad]
            'FR_foot_joint': 0. ,  # [rad]
            'RR_foot_joint': 0.,   # [rad]   
        }

    class env(LeggedRobotCfg.env):
        num_envs = 6144
        num_privileged_group = 4096
        num_proprio_group = 2048
        num_observations = 58
        num_privileged_obs = 247 - 4 + 12 + 16 + 5# 57 + 187 + 3
        num_actions = 16
        num_actions_leg = 12
        num_actions_wheel = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        max_init_terrain_level = 3 # starting curriculum state

    class domain_rand( LeggedRobotCfg.domain_rand ):
        enforce_left_right_symmetry = True
        push_robots = True
        push_interval_s = 15.0
        max_push_vel_xy = 1.5

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        base_com_range = [-0.05, 0.05]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2]  

        randomize_legMotor_zero_offset = True
        legMotor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        randomize_joint_friction = True
        joint_friction_range = [0.01, 0.1]

        randomize_joint_stiffness = True
        joint_stiffness_range = [0.0, 0.0]

        randomize_joint_damping = True
        joint_damping_range = [0.001, 0.1]

        randomize_joint_armature = True
        joint_armature_range = [0.0001, 0.005]     # range to contain the real joint armature 

        # 新增的延迟随机化
        add_obs_latency = True
        randomize_obs_motor_latency = True
        range_obs_motor_latency = [0, 2]
        randomize_obs_imu_latency = True
        range_obs_imu_latency = [0, 4]

        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 2]

    class control( LeggedRobotCfg.control ):
        control_type = 'pAndV'      # position control for joint + velocity control for wheel
        stiffnessLeg = {
            'FL_hip_joint': 20.,    # [N*m/rad]
            'RL_hip_joint': 20.,
            'FR_hip_joint': 20.,
            'RR_hip_joint': 20.,             

            'FL_thigh_joint': 20.,   
            'RL_thigh_joint': 20., 
            'FR_thigh_joint': 20.,   
            'RR_thigh_joint': 20., 

            'FL_calf_joint': 20., 
            'RL_calf_joint': 20.,  
            'FR_calf_joint': 20.,
            'RR_calf_joint': 20.,  
        }
        dampingLeg = {
            'FL_hip_joint': 0.5,    # [N*m*s/rad]
            'RL_hip_joint': 0.5,
            'FR_hip_joint': 0.5,
            'RR_hip_joint': 0.5,             

            'FL_thigh_joint': 0.5,   
            'RL_thigh_joint': 0.5, 
            'FR_thigh_joint': 0.5,   
            'RR_thigh_joint': 0.5, 

            'FL_calf_joint': 0.5, 
            'RL_calf_joint': 0.5,  
            'FR_calf_joint': 0.5,
            'RR_calf_joint': 0.5,
        }
        stiffnessWheel = {
            'FL_foot_joint': 0.8, 
            'RL_foot_joint': 0.8, 
            'FR_foot_joint': 0.8,
            'RR_foot_joint': 0.8,
        }
        dampingWheel = {
            'FL_foot_joint': 1e-4, 
            'RL_foot_joint': 1e-4, 
            'FR_foot_joint': 1e-4,
            'RR_foot_joint': 1e-4,    
        }
        action_scale_pos = 0.25
        action_scale_vel = 2.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w_description/urdf/go2w_description.urdf'
        name = "go2w"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        wheelRadius = 0.088  # meters
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.38 # 0.391.2
        only_positive_rewards = False
        gait_freq = 2.0  # Hz
        gait_offset_leftward = [0.5, 0.0, 0.0, 0.5]  # FL, FR, RL, RR
        gait_offset_rightward = [0.0, 0.5, 0.5, 0.0]  # FL, FR, RL, RR
        contact_shaping_sigma = 50.0
        class scales( LeggedRobotCfg.rewards.scales ):
            collision = -1.
            dof_acc_leg = -2.5e-7
            dof_acc_wheel = -1e-7
            action_rate_leg = -0.01
            action_rate_wheel = -0.0001
            torques = -0.0002
            joint_power = -2e-7
            dof_pos_leg_limits = -10.0
        
            tracking_lin_vel = 4.0
            tracking_ang_vel = 2.0
            alive = 0.5
            feet_stumble = -0.5
            tracking_gait = 2.0

            base_height = 0.5 # new reward for base height, -1 ~ 0
            orientation = -0.01
            lin_vel_z = -0.5
            ang_vel_xy = -0.05

            wheel_air = -0.0  # new reward to penalize wheel air when no y command, -inf ~ 0
            feet_air_time =  0.0
            encourage_wheel_for_x = -0.00 # new reward to encourage wheel rotation for forward x vel, 0~2
            stand_still = -0.0
            action_smoothness = -0.0
            action_rate = -0.00
            dof_acc = -0.0
            same_wheel_x_position = -0.0  # new reward to encourage same x position for left/right wheel pairs, -1 ~ 0


class GO2wRoughCfgPPO( LeggedRobotCfgPPO ):  
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01 # 0.01

    class proprioceptive_encoder:
        proprioceptive_encoder_input_dim = GO2wRoughCfg.env.num_observations * GO2wRoughCfg.env.num_obs_hist
        num_latent_dim = 32
        proprioceptive_encoder_hidden_dims = [512, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_normalize = 1

    class privileged_encoder:
        privileged_encoder_input_dim = GO2wRoughCfg.env.num_privileged_obs
        num_latent_dim = 32
        PrivilegedEnc_hidden_dims = [512, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_normalize = 1

    class runner:
        num_steps_per_env = 24 # per iteration
        max_iterations = 30000 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'go2w_plane'
        run_name = ''
        empirical_normalization = False
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt