from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class M20RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        dof_idx_leg = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14] # FL -> FR -> RL -> RR
        dof_idx_wheel = [3, 7, 11, 15]
        pos = [0.0, 0.0, 0.55] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'fl_hipx_joint': -0.1,   # [rad]
            'hl_hipx_joint': -0.1,   # [rad]
            'fr_hipx_joint': 0.1 ,  # [rad]
            'hr_hipx_joint': 0.1,   # [rad]            

            'fl_hipy_joint': -0.65,     # [rad]
            'hl_hipy_joint': 0.65,   # [rad]
            'fr_hipy_joint': -0.65,     # [rad]
            'hr_hipy_joint': 0.65,   # [rad]

            'fl_knee_joint': 1.5,   # [rad]
            'hl_knee_joint': -1.5,    # [rad]
            'fr_knee_joint': 1.5,  # [rad]
            'hr_knee_joint': -1.5,    # [rad]

            'fl_wheel_joint': 0.,   # [rad]
            'hl_wheel_joint': 0.,   # [rad]
            'fr_wheel_joint': 0. ,  # [rad]
            'hr_wheel_joint': 0.,   # [rad]   
        }

    class env(LeggedRobotCfg.env):
        num_envs = 6144
        num_privileged_group = 4096
        num_proprio_group = 2048
        num_observations = 53
        num_privileged_obs = 247 - 4 + 12 + 16 # 57 + 187 + 3
        num_actions = 16
        num_actions_leg = 12
        num_actions_wheel = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        max_init_terrain_level = 1 # starting curriculum state
        num_rows= 20 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]

    class domain_rand( LeggedRobotCfg.domain_rand ):
        # 当启用 symmetry（数据增强/镜像损失）时，建议打开该开关，
        # 让与关节相关的随机化（目前 PD 增益 + 零点偏置 + CG y 偏移 + link）在左右腿之间保持一致。
        # 否则镜像样本不再满足等价动力学/奖励，容易在训练后期出现抖动/乱动。
        enforce_left_right_symmetry = True
        left_body_index = [1, 2, 3, 4, 9, 10, 11, 12] # debug 看 body_name
        right_body_index = [5, 6, 7, 8, 13, 14, 15, 16]

        push_robots = True
        push_interval_s = 15.0
        max_push_vel_xy = 1.5

        randomize_base_mass = True
        added_mass_range = [-1., 20.]

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
        joint_friction_range = [0.0, 0.1]#[0.01, 0.1]

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
        stiffnessLeg = {
            'fl_hipx_joint': 80.,    # [N*m/rad]
            'hl_hipx_joint': 80.,
            'fr_hipx_joint': 80.,
            'hr_hipx_joint': 80.,             

            'fl_hipy_joint': 80.,   
            'hl_hipy_joint': 80., 
            'fr_hipy_joint': 80.,   
            'hr_hipy_joint': 80., 

            'fl_knee_joint': 80., 
            'hl_knee_joint': 80.,  
            'fr_knee_joint': 80.,
            'hr_knee_joint': 80.,  
        }
        dampingLeg = {
            'fl_hipx_joint': 2.,    # [N*m*s/rad]
            'hl_hipx_joint': 2.,
            'fr_hipx_joint': 2.,
            'hr_hipx_joint': 2.,             

            'fl_hipy_joint': 2.,   
            'hl_hipy_joint': 2., 
            'fr_hipy_joint': 2.,   
            'hr_hipy_joint': 2., 

            'fl_knee_joint': 2., 
            'hl_knee_joint': 2.,  
            'fr_knee_joint': 2.,
            'hr_knee_joint': 2.,
        }
        stiffnessWheel = {
            'fl_wheel_joint': 0., 
            'hl_wheel_joint': 0., 
            'fr_wheel_joint': 0.,
            'hr_wheel_joint': 0.,
        }
        dampingWheel = {
            'fl_wheel_joint': 0.6, 
            'hl_wheel_joint': 0.6, 
            'fr_wheel_joint': 0.6,
            'hr_wheel_joint': 0.6,
        }
        action_scale_pos = 0.25
        action_scale_vel = 10.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/M20_urdf/urdf/M20.urdf'
        name = "M20"
        foot_name = "wheel"
        penalize_contacts_on = ["hipx", "hipy", "knee"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        # wheelRadius = 0.088  # meters
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 1.0
        base_height_target = 0.456 + 0.05
        only_positive_rewards = False
        gait_freq = 2  # Hz
        gait_offset_leftward = [0.5, 0.0, 0.0, 0.5]  # FL, FR, RL, RR
        gait_offset_rightward = [0.0, 0.5, 0.5, 0.0]  # FL, FR, RL, RR
        contact_shaping_sigma = 50.0

        # Gait tracking reward shaping (phase/contact schedule)
        isGaitInput = False
        gait_swing_height_target = 0.2 # 0.16
        gait_stance_vel_gamma = 5.0
        gait_w_contact = 1.0
        gait_w_swing_height = 0.5
        gait_w_stance_vel = 0.25

        class scales():
            collision = -1.
            dof_acc_leg = -1e-7
            # dof_acc_wheel = -1e-8
            action_rate_leg = -5e-3
            action_rate_wheel = -1e-4
            # torques = -0.0002
            joint_power = -2e-7
            # dof_pos_leg_limits = -10.0
        
            tracking_lin_vel = 4.0
            tracking_ang_vel = 2.0
            # tracking_xgait = 0.5
            # alive = 0.5
            termination = -100.0
            # zCmd_motion = -1.0
            zCmd_wheel_vel = -5e-3
            # zCmd_pos_dist = -10.0
            hip_pos = -50.0
            hip_vel = 1.0
            knee_joint = -0.5
            stand_still = -5.0
            # action_amplitude = -0.001
            same_wheel_x_position = -5.0
            feet_stumble = -0.05
            feet_air_time = 2.0
            unnec_leg_lift = -1.0

            base_height = 5 # new reward for base height, -1 ~ 0
            orientation = -10.0
            lin_vel_z = -0.5 * 2
            ang_vel_xy = -0.2


class M20RoughCfgPPO( LeggedRobotCfgPPO ):  
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005 # 0.01
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = False
            use_mirror_critic_loss = True
            mirror_loss_coeff = 2.0
            mirror_critic_loss_coeff = 1.0
            data_mirror_func = "legged_gym.utils.symmetry_cts:go2w_symmetry"

    class proprioceptive_encoder:
        proprioceptive_encoder_input_dim = M20RoughCfg.env.num_observations * M20RoughCfg.env.num_obs_hist
        num_est_vel_dim = 3
        num_latent_dim = 29
        # proprioceptive_encoder_hidden_dims = [512, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid100
        output_normalize = 1
        normalize_latent = False

    class privileged_encoder:
        privileged_encoder_input_dim = M20RoughCfg.env.num_privileged_obs
        num_est_vel_dim = 3
        num_latent_dim = 29
        # PrivilegedEnc_hidden_dims = [512, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_normalize = 1
        normalize_latent = False

    class runner:
        num_steps_per_env = 24 # per iteration
        max_iterations = 30000 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'm20_terrain'
        run_name = ''
        empirical_normalization = False
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = "{LEGGED_GYM_ROOT_DIR}/logs/model/m20_plane/model_600.pt" # updated from load_run and chkpt
        resume_path = "{LEGGED_GYM_ROOT_DIR}/logs/m20_terrain/Mar20_20-13-02_/model_15000.pt" # seed 2 往右倾斜？
        # resume_path = None