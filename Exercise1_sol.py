import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import imageio

import gymnasium as gym
from src.EA.ES import ES, ES_opts
from src.world.robot.controllers import MLP
from src.utils.Filesys import get_project_root
from src.world.World import World

ROOT_DIR = get_project_root()
ENV_NAME = 'Hopper_custom' 
XML_FILE  = os.path.join(ROOT_DIR, "src/world/robot/assets/hopper_kangaroo.xml")


class HopperWorld(World):
    def __init__(self):
        self.env = gym.make(ENV_NAME, xml_file=XML_FILE)#, render_mode="human")
        action_space = self.env.action_space.shape[0]
        state_space  = self.env.observation_space.shape[0]
        self.controller = MLP.NNController(state_space, action_space)
        #self.controller = MLP.NN_najaroController(state_space, action_space)
        self.dt     = self.env.get_wrapper_attr('dt')
        self.n_params = self.controller.n_params

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)
        return self.controller
    
    def evaluate_individual(self, genotype):
        trial_time = 10  # seconds in simulation
        n_sim_steps = int(trial_time / self.dt)

        self.geno2pheno(genotype)

        rewards_list = []
        observations, info = self.env.reset(seed=42)
        for step in range(n_sim_steps):
            action = self.controller.get_action(observations)
            observations, rewards, terminated, truncated, info = self.env.step(action)
            rewards_list.append(rewards)
        return np.sum(rewards_list)

    """ def evaluate_individual(self, genotype):
        
        Fitness =
            8  · jump_distance
          + 0.5· jump_height
          + 2  · air_time
          + 1  · ∫|hip + knee| dt        (bonus)
          - 1  · ∫|ankle τ| dt            (penalty)
          (ignore energy_penalty by default)
        
        # --- hyper-params ---
        trial_time      = 50.0
        contact_eps     = 1e-2
        distance_gain   = 5.0
        height_gain     = 0.5
        airtime_gain    = 2.0
        energy_penalty  = 0.0

        EXTENSION_GAIN  = 1.0   # bonus hip+knee
        ANKLE_TAX       = 1.0   # penalty ankle-only

        # -------------------
        n_steps = int(trial_time / self.dt)

        self.geno2pheno(genotype)
        obs, _ = self.env.reset()
        sim    = self.env.unwrapped
        model  = sim.model
        data   = sim.data

        # resolve IDs once
        hip_jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thigh_joint")
        knee_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "leg_joint")
        ankle_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "foot_joint")

        hip_qadr  = model.jnt_qposadr[hip_jid]   if hip_jid  != -1 else None
        knee_qadr = model.jnt_qposadr[knee_jid]  if knee_jid != -1 else None

        x0, z0   = data.qpos[0], data.qpos[1]
        max_z    = z0
        takeoff  = None
        landing  = None
        ext_bonus = 0.0
        ankle_cost = 0.0
        energy_used = 0.0

        for step in range(n_steps):
            action = self.controller.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)

            # track height
            z = data.qpos[1]
            max_z = max(max_z, z)

            # detect ground contact (skip root body)
            on_ground = (abs(data.cfrc_ext[1:, 2]) > contact_eps).any()

            if takeoff is None and not on_ground:
                takeoff = step
            elif takeoff is not None and on_ground:
                landing = step
                break

            # extension bonus
            if hip_qadr is not None:
                ext_bonus += abs(data.qpos[hip_qadr]) * self.dt
            if knee_qadr is not None:
                ext_bonus += abs(data.qpos[knee_qadr]) * self.dt

            # ankle penalty
            if ankle_aid != -1:
                ankle_cost += abs(data.actuator_force[ankle_aid]) * self.dt

            # total energy (if you ever want to enable)
            energy_used += abs(data.actuator_force).sum() * self.dt

            if terminated or truncated:
                break

        # summary
        x_end        = data.qpos[0]
        jump_distance = max(0.0, x_end - x0)

        air_time = 0.0
        if takeoff is not None and landing is not None:
            air_time = (landing - takeoff) * self.dt

        jump_height = max(0.0, max_z - z0)

        score = (
            distance_gain  * jump_distance
          + height_gain    * jump_height
          + airtime_gain   * air_time
          + EXTENSION_GAIN * ext_bonus
          #- ANKLE_TAX      * ankle_cost
          - energy_penalty * energy_used
        )
        return score """


def run_EA(ea: ES, world: HopperWorld):
    """
    Runs the ES for ea.n_gen generations on the given world.
    Returns:
        best_hist – np.ndarray of length n_gen (best-of-pop fitness per gen)
        mean_hist – np.ndarray of length n_gen (mean fitness per gen)
    Also saves populations inside ea, etc., according to EA’s own logging.
    """
    best_hist = np.zeros(ea.n_gen, dtype=float)
    mean_hist = np.zeros(ea.n_gen, dtype=float)

    for g in range(ea.n_gen):
        pop = ea.ask()
        fitnesses = np.zeros(ea.n_pop, dtype=float)

        for i, geno in enumerate(pop):
            fitnesses[i] = world.evaluate_individual(geno)

        ea.tell(pop, fitnesses)

        best_hist[g] = fitnesses.max()
        mean_hist[g] = fitnesses.mean()
        
        if g % 10 == 0:
            best_idx = np.argmax(fitnesses)
            best_genotype = pop[best_idx]
            world.controller.geno2pheno(best_genotype)
            results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'ES')
            video_path = os.path.join(results_dir, f"video_gen{g}.mp4")
            generate_best_individual_video(world.controller, video_path)

    return best_hist, mean_hist


def plot_es_fitness(best_hist, mean_hist=None, title="ES learning curve"):
    """
    Draw best‐of‐pop (and optional mean) vs. generation.
    """
    gens = np.arange(len(best_hist))
    plt.figure(figsize=(6, 4))
    plt.plot(gens, best_hist, label="best of pop")
    if mean_hist is not None:
        plt.plot(gens, mean_hist, "--", label="mean of pop", alpha=0.6)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def record_trajectory(env_name: str,
                    xml_file: str,
                    controller,
                    duration: float = 30.0,
                    render: bool = False):
    """
    Run the given controller once and return a dict with trajectory data.
    Fixed to properly handle body positions and joint angles.
    """
    env = gym.make(env_name, xml_file=xml_file, render_mode="rgb_array" if render else None)
    sim = env.unwrapped
    mdl = sim.model
    data = sim.data
    dt = env.get_wrapper_attr('dt')
    T = int(duration / dt)

    # Initialize arrays with proper size based on actual simulation steps
    max_steps = T
    t_list = []
    com_list = []
    hip_p_list = []
    knee_p_list = []
    ankle_p_list = []
    joint_q_list = []
    joint_tau_list = []

    # Resolve body IDs for position tracking
    try:
        torso_body = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, "torso")
        thigh_body = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, "thigh") 
        leg_body = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, "leg")
        foot_body = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, "foot")
    except:
        # Fallback if body names are different
        torso_body = 1  # Usually torso is body 1
        thigh_body = 2
        leg_body = 3
        foot_body = 4

    obs, _ = env.reset()
    
    for k in range(max_steps):
        action = controller.get_action(obs)
        obs, _, term, trunc, _ = env.step(action)

        # Store time
        t_list.append(k * dt)

        # Extract positions from MuJoCo data
        # For hopper, we need to get the actual body positions
        try:
            # Center of mass of root body (torso)
            com_pos = data.subtree_com[0, [0, 2]]  # x, z coordinates
            
            # Body positions - extract x,z coordinates  
            hip_pos = data.xpos[thigh_body, [0, 2]]    # thigh body position
            knee_pos = data.xpos[leg_body, [0, 2]]     # leg body position  
            ankle_pos = data.xpos[foot_body, [0, 2]]   # foot body position
            
        except:
            # Fallback: use joint positions to estimate body positions
            # This is an approximation based on the kinematic chain
            root_x, root_z = data.qpos[0], data.qpos[1] if len(data.qpos) > 1 else (0, 1.25)
            
            com_pos = np.array([root_x, root_z])
            hip_pos = np.array([root_x, root_z - 0.05])
            knee_pos = np.array([root_x + 0.05, root_z - 0.4])  
            ankle_pos = np.array([root_x + 0.1, root_z - 0.75])

        com_list.append(com_pos)
        hip_p_list.append(hip_pos)
        knee_p_list.append(knee_pos)
        ankle_p_list.append(ankle_pos)

        # Joint angles from observation space
        # Based on the observation description: indices 2,3,4 are thigh, leg, foot joint angles
        if len(obs) >= 5:
            joint_angles = obs[2:5]  # thigh_joint, leg_joint, foot_joint
        else:
            joint_angles = np.array([0, 0, 0])
            
        joint_q_list.append(joint_angles)

        # Joint torques from actuator forces
        try:
            joint_torques = data.actuator_force[:3] if len(data.actuator_force) >= 3 else np.array([0, 0, 0])
        except:
            joint_torques = np.array([0, 0, 0])
            
        joint_tau_list.append(joint_torques)

        if term or trunc:
            break

    env.close()
    
    # Convert lists to numpy arrays
    actual_steps = len(t_list)
    t = np.array(t_list)
    com = np.array(com_list)
    hip_p = np.array(hip_p_list) 
    knee_p = np.array(knee_p_list)
    ankle_p = np.array(ankle_p_list)
    joint_q = np.array(joint_q_list)
    joint_tau = np.array(joint_tau_list)
    
    return dict(t=t, com=com,
                hip=hip_p, knee=knee_p, ankle=ankle_p,
                joint_q=joint_q, joint_tau=joint_tau,
                dt=dt, actual_duration=t[-1] if len(t) > 0 else 0)


def plot_joint_angles(traj, title="Joint angles vs time"):
    """
    Shows hip, knee, ankle angles (deg) for *one* run.
    Fixed to handle variable trajectory length and proper joint naming.
    """
    t = traj['t']
    if len(t) == 0:
        print("No trajectory data to plot!")
        return
        
    ang = np.rad2deg(traj['joint_q'])
    
    plt.figure(figsize=(10, 6))
    joint_names = ["Hip (thigh)", "Knee (leg)", "Ankle (foot)"]
    colors = ['blue', 'red', 'green']
    
    for j, (name, color) in enumerate(zip(joint_names, colors)):
        if j < ang.shape[1]:  # Make sure we have data for this joint
            plt.plot(t, ang[:, j], label=name, color=color, linewidth=2)
    
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [degrees]")
    plt.title(f"{title} (Duration: {traj.get('actual_duration', t[-1] if len(t) > 0 else 0):.2f}s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_leg_xy(traj, title="Sagittal-plane leg path"):
    """
    Draws the XY path of ankle, knee, hip and CoM for *one* run.
    Fixed to handle proper coordinate system and trajectory length.
    """
    if len(traj['t']) == 0:
        print("No trajectory data to plot!")
        return
        
    plt.figure(figsize=(10, 7))
    
    # Plot trajectories with different styles
    plt.plot(traj['ankle'][:, 0], traj['ankle'][:, 1], 'o-', label="Ankle (foot)", 
             color='green', markersize=3, alpha=0.7)
    plt.plot(traj['knee'][:, 0], traj['knee'][:, 1], 's-', label="Knee (leg)", 
             color='red', markersize=3, alpha=0.7)
    plt.plot(traj['hip'][:, 0], traj['hip'][:, 1], '^-', label="Hip (thigh)", 
             color='blue', markersize=3, alpha=0.7)
    plt.plot(traj['com'][:, 0], traj['com'][:, 1], 'D-', label="Center of Mass", 
             color='black', markersize=4, linewidth=2)
    
    # Mark start and end points
    for name, data, color in [('ankle', traj['ankle'], 'green'), 
                             ('knee', traj['knee'], 'red'),
                             ('hip', traj['hip'], 'blue'),
                             ('com', traj['com'], 'black')]:
        plt.plot(data[0, 0], data[0, 1], 'o', color=color, markersize=8, 
                markeredgecolor='white', markeredgewidth=2, label=f'{name.title()} start')
        plt.plot(data[-1, 0], data[-1, 1], 'x', color=color, markersize=10, 
                markeredgewidth=3, label=f'{name.title()} end')
    
    plt.xlabel("X position [m]")
    plt.ylabel("Z position [m]") 
    plt.title(f"{title} (Duration: {traj.get('actual_duration', 0):.2f}s)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def generate_best_individual_video(controller, video_name: str = 'EvoRob1_video.mp4'):
    # TODO: Make a video of the best individual, and plot the fitness curve.
    env = gym.make(ENV_NAME, xml_file=XML_FILE, render_mode="rgb_array")
    rewards_list = []
    observations, info = env.reset()
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()


def main():
    """
    Updated main function using the corrected functions.
    """

    world = HopperWorld()
    n_parameters = world.n_params

    ES_opts["min"] = -1
    ES_opts["max"] = 1
    ES_opts["num_parents"] = 16
    ES_opts["num_generations"] = 100
    ES_opts["mutation_sigma"] = 0.7

    population_size = 80

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'ES')
    ea = ES(population_size, n_parameters, ES_opts, results_dir)

    # Use the corrected run_EA function
    best_hist, mean_hist = run_EA(ea, world)   
    #plot_es_fitness(best_hist, mean_hist)              
    # Load and test best individual
    best_individual = np.load(os.path.join(results_dir, f"{ES_opts['num_generations']-1}", "x_best.npy"))
    world.controller.geno2pheno(best_individual)
    video_path = os.path.join(results_dir, f"video_gen{ES_opts['num_generations']}.mp4")
    generate_best_individual_video(world.controller, video_path)


if __name__ == '__main__':
    main()