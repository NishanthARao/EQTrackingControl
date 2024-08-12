import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import sys
import ast

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

## Usage: python3 make_plots env_name
## Eg.,   python3 make_plots test_random_walk_velocity

# Load data
# The named variables (keys) are ["mean_rewards", "std_rewards", "mean_terminal_timesteps", "std_terminal_timesteps", "x_vals_rew", "x_vals_ts"]
# non_eq_data_path = np.load("./plots/ppo_jax_3_layer_no_eq.npz")
# eq_data_path = np.load("./plots/ppo_jax_3_layer_eq.npz")

# non_eq_data_path = "./checkpoints/position_no_eq_small_noise_tanh_final_activation_noise_001_gamma_1/training_data.npz"
# eq_data_path = "./checkpoints/position_eq_small_noise_tanh_final_activation_noise_001_gamma_1_4/training_data.npz"

env_name = sys.argv[1]

data_dir = "./checkpoints/"

non_eq_data_path = data_dir + env_name + "/training_data.npz"
p_eq_data_path = data_dir + env_name + "_p_equivariant/training_data.npz"
v_eq_data_path = data_dir + env_name + "_v_equivariant/training_data.npz"
pv_eq_data_path = data_dir + env_name + "_pv_equivariant/training_data.npz"
pva_eq_data_path = data_dir + env_name + "_pva_equivariant/training_data.npz"

# eq_data_path = "./checkpoints/constant_velocity_eq_50M_1/training_data.npz"
# non_eq_data_path = "./checkpoints/constant_velocity_no_eq_50M/training_data.npz"

p_eq_data = np.load(p_eq_data_path)
v_eq_data = np.load(v_eq_data_path)
pv_eq_data = np.load(pv_eq_data_path)
non_eq_data = np.load(non_eq_data_path)
try:
    pva_eq_data = np.load(pva_eq_data_path)
except:
    pva_eq_data_path = None

p_eq_args = None
with open(os.path.dirname(p_eq_data_path) + "/config.txt", "r") as f:
    p_eq_args = ast.literal_eval(f.read())

v_eq_args = None
with open(os.path.dirname(v_eq_data_path) + "/config.txt", "r") as f:
    v_eq_args = ast.literal_eval(f.read())

pv_eq_args = None
with open(os.path.dirname(pv_eq_data_path) + "/config.txt", "r") as f:
    pv_eq_args = ast.literal_eval(f.read())

# Plot data
plt.figure()
fig, ax = plt.subplots()
# axins = zoomed_inset_axes(ax, 8, loc="center right")

ax.plot(non_eq_data["x_vals_rew"], non_eq_data["mean_rewards"], label="Non-Equivariant")
ax.fill_between(non_eq_data["x_vals_rew"], non_eq_data["mean_rewards"] - non_eq_data["std_rewards"], non_eq_data["mean_rewards"] + non_eq_data["std_rewards"], alpha=0.5)
ax.plot(p_eq_data["x_vals_rew"], p_eq_data["mean_rewards"], label="P-Equivariant")
ax.fill_between(p_eq_data["x_vals_rew"], p_eq_data["mean_rewards"] - p_eq_data["std_rewards"], p_eq_data["mean_rewards"] + p_eq_data["std_rewards"], alpha=0.5)
ax.plot(v_eq_data["x_vals_rew"], v_eq_data["mean_rewards"], label="V-Equivariant")
ax.fill_between(v_eq_data["x_vals_rew"], v_eq_data["mean_rewards"] - v_eq_data["std_rewards"], v_eq_data["mean_rewards"] + v_eq_data["std_rewards"], alpha=0.5)
ax.plot(pv_eq_data["x_vals_rew"], pv_eq_data["mean_rewards"], label="PV-Equivariant")
ax.fill_between(pv_eq_data["x_vals_rew"], pv_eq_data["mean_rewards"] - pv_eq_data["std_rewards"], pv_eq_data["mean_rewards"] + pv_eq_data["std_rewards"], alpha=0.5)

if pva_eq_data_path is not None:
    ax.plot(pva_eq_data["x_vals_rew"], pva_eq_data["mean_rewards"], label="PVA-Equivariant")
    ax.fill_between(pva_eq_data["x_vals_rew"], pva_eq_data["mean_rewards"] - pva_eq_data["std_rewards"], pva_eq_data["mean_rewards"] + pva_eq_data["std_rewards"], alpha=0.5)

ax.grid(True)
ax.legend(loc="best")
ax.set_xlabel("Env Steps")
ax.set_ylabel("Mean Reward")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x/1e6)}M"))
ax.set_title(f"Reward Curve: {p_eq_args['env_name']} \n PPO JAX 3 Layer Policy Averaged over {p_eq_args['num_seeds']} Seeds \n {p_eq_args['NUM_ENVS']} Envs {p_eq_args['NUM_STEPS']} Steps {p_eq_args['TOTAL_TIMESTEPS']:.2E} Steps")
# sub region of the original image
# x1, x2, y1, y2 = 4e7, 5e7, -5, 0.01
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.set_yticks([-4, -2, 0])
# axins.plot(non_eq_data["x_vals_rew"], non_eq_data["mean_rewards"], label="Non-Equivariant")
# axins.fill_between(non_eq_data["x_vals_rew"], non_eq_data["mean_rewards"] - non_eq_data["std_rewards"], non_eq_data["mean_rewards"] + non_eq_data["std_rewards"], alpha=0.5)
# axins.plot(eq_data["x_vals_rew"], eq_data["mean_rewards"], label="Equivariant")
# axins.fill_between(eq_data["x_vals_rew"], eq_data["mean_rewards"] - eq_data["std_rewards"], eq_data["mean_rewards"] + eq_data["std_rewards"], alpha=0.5)
# axins.grid(True)
# axins.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{(x/1e6):.1f}M"))

# # draw a bbox of the region of the inset axes in the parent axes and connecting lines between the bbox and the inset axes area
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.savefig(data_dir + env_name + "_eq_vs_no_eq_rewards.png", dpi=1000)


plt.figure()
plt.plot(non_eq_data["x_vals_ts"], non_eq_data["mean_terminal_timesteps"], label="Non-Equivariant")
plt.fill_between(non_eq_data["x_vals_ts"], non_eq_data["mean_terminal_timesteps"] - non_eq_data["std_terminal_timesteps"], non_eq_data["mean_terminal_timesteps"] + non_eq_data["std_terminal_timesteps"], alpha=0.5)
plt.plot(p_eq_data["x_vals_ts"], p_eq_data["mean_terminal_timesteps"], label="P-Equivariant")
plt.fill_between(p_eq_data["x_vals_ts"], p_eq_data["mean_terminal_timesteps"] - p_eq_data["std_terminal_timesteps"], p_eq_data["mean_terminal_timesteps"] + p_eq_data["std_terminal_timesteps"], alpha=0.5)
plt.plot(v_eq_data["x_vals_ts"], v_eq_data["mean_terminal_timesteps"], label="V-Equivariant")
plt.fill_between(v_eq_data["x_vals_ts"], v_eq_data["mean_terminal_timesteps"] - v_eq_data["std_terminal_timesteps"], v_eq_data["mean_terminal_timesteps"] + v_eq_data["std_terminal_timesteps"], alpha=0.5)
plt.plot(pv_eq_data["x_vals_ts"], pv_eq_data["mean_terminal_timesteps"], label="PV-Equivariant")
plt.fill_between(pv_eq_data["x_vals_ts"], pv_eq_data["mean_terminal_timesteps"] - pv_eq_data["std_terminal_timesteps"], pv_eq_data["mean_terminal_timesteps"] + pv_eq_data["std_terminal_timesteps"], alpha=0.5)

if pva_eq_data_path is not None:
    plt.plot(pva_eq_data["x_vals_ts"], pva_eq_data["mean_terminal_timesteps"], label="PVA-Equivariant")
    plt.fill_between(pva_eq_data["x_vals_ts"], pva_eq_data["mean_terminal_timesteps"] - pva_eq_data["std_terminal_timesteps"], pva_eq_data["mean_terminal_timesteps"] + pva_eq_data["std_terminal_timesteps"], alpha=0.5)

plt.xlabel("Env Steps")
plt.ylabel("Mean Terminal Timesteps")
plt.grid(True)
plt.legend(loc="best")
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x/1e6)}M"))
plt.title(f"Agent Timesteps: {p_eq_args['env_name']} \n PPO JAX 3 Layer Policy Averaged over {p_eq_args['num_seeds']} Seeds \n {p_eq_args['NUM_ENVS']} Envs {p_eq_args['NUM_STEPS']} Steps {p_eq_args['TOTAL_TIMESTEPS']:.2E} Steps")
plt.savefig(data_dir + env_name + "_eq_vs_no_eq_timesteps.png", dpi=1000)


# Compare reward plots
non_eq_pos_data_path = data_dir + env_name + "/pos_data.npy"
non_eq_ref_pos_data_path = data_dir + env_name + "/ref_pos_data.npy"
non_eq_vel_data_path = data_dir + env_name + "/vel_data.npy"
non_eq_ref_vel_data_path = data_dir + env_name + "/ref_vel_data.npy"

p_eq_pos_data_path = data_dir + env_name + "_p_equivariant/pos_data.npy"
p_eq_ref_pos_data_path = data_dir + env_name + "_p_equivariant/ref_pos_data.npy"
p_eq_vel_data_path = data_dir + env_name + "_p_equivariant/vel_data.npy"
p_eq_ref_vel_data_path = data_dir + env_name + "_p_equivariant/ref_vel_data.npy"

v_eq_pos_data_path = data_dir + env_name + "_v_equivariant/pos_data.npy"
v_eq_ref_pos_data_path = data_dir + env_name + "_v_equivariant/ref_pos_data.npy"
v_eq_vel_data_path = data_dir + env_name + "_v_equivariant/vel_data.npy"
v_eq_ref_vel_data_path = data_dir + env_name + "_v_equivariant/ref_vel_data.npy"

pv_eq_pos_data_path = data_dir + env_name + "_pv_equivariant/pos_data.npy"
pv_eq_ref_pos_data_path = data_dir + env_name + "_pv_equivariant/ref_pos_data.npy"
pv_eq_vel_data_path = data_dir + env_name + "_pv_equivariant/vel_data.npy"
pv_eq_ref_vel_data_path = data_dir + env_name + "_pv_equivariant/ref_vel_data.npy"

pva_eq_pos_data_path = data_dir + env_name + "_pva_equivariant/pos_data.npy"
pva_eq_ref_pos_data_path = data_dir + env_name + "_pva_equivariant/ref_pos_data.npy"
pva_eq_vel_data_path = data_dir + env_name + "_pva_equivariant/vel_data.npy"
pva_eq_ref_vel_data_path = data_dir + env_name + "_pva_equivariant/ref_vel_data.npy"

non_eq_pos_data = np.load(non_eq_pos_data_path)
non_eq_ref_pos_data = np.load(non_eq_ref_pos_data_path)
non_eq_vel_data = np.load(non_eq_vel_data_path)
non_eq_ref_vel_data = np.load(non_eq_ref_vel_data_path)

p_eq_pos_data = np.load(p_eq_pos_data_path)
p_eq_ref_pos_data = np.load(p_eq_ref_pos_data_path)
p_eq_vel_data = np.load(p_eq_vel_data_path)
p_eq_ref_vel_data = np.load(p_eq_ref_vel_data_path)

v_eq_pos_data = np.load(v_eq_pos_data_path)
v_eq_ref_pos_data = np.load(v_eq_ref_pos_data_path)
v_eq_vel_data = np.load(v_eq_vel_data_path)
v_eq_ref_vel_data = np.load(v_eq_ref_vel_data_path)

pv_eq_pos_data = np.load(pv_eq_pos_data_path)
pv_eq_ref_pos_data = np.load(pv_eq_ref_pos_data_path)
pv_eq_vel_data = np.load(pv_eq_vel_data_path)
pv_eq_ref_vel_data = np.load(pv_eq_ref_vel_data_path)

try:
    pva_eq_pos_data = np.load(pva_eq_pos_data_path)
    pva_eq_ref_pos_data = np.load(pva_eq_ref_pos_data_path)
    pva_eq_vel_data = np.load(pva_eq_vel_data_path)
    pva_eq_ref_vel_data = np.load(pva_eq_ref_vel_data_path)
except:
    pass

non_eq_pos_errors = np.linalg.norm(non_eq_pos_data - non_eq_ref_pos_data, axis=-1)
non_eq_mean_pos_errors = np.mean(non_eq_pos_errors, axis=1)
non_eq_std_pos_errors = np.std(non_eq_pos_errors, axis=1)

p_eq_pos_errors = np.linalg.norm(p_eq_pos_data - p_eq_ref_pos_data, axis=-1)
p_eq_mean_pos_errors = np.mean(p_eq_pos_errors, axis=1)
p_eq_std_pos_errors = np.std(p_eq_pos_errors, axis=1)

v_eq_pos_errors = np.linalg.norm(v_eq_pos_data - v_eq_ref_pos_data, axis=-1)
v_eq_mean_pos_errors = np.mean(v_eq_pos_errors, axis=1)
v_eq_std_pos_errors = np.std(v_eq_pos_errors, axis=1)

pv_eq_pos_errors = np.linalg.norm(pv_eq_pos_data - pv_eq_ref_pos_data, axis=-1)
pv_eq_mean_pos_errors = np.mean(pv_eq_pos_errors, axis=1)
pv_eq_std_pos_errors = np.std(pv_eq_pos_errors, axis=1)

try:
    pva_eq_pos_errors = np.linalg.norm(pva_eq_pos_data - pva_eq_ref_pos_data, axis=-1)
    pva_eq_mean_pos_errors = np.mean(pva_eq_pos_errors, axis=1)
    pva_eq_std_pos_errors = np.std(pva_eq_pos_errors, axis=1)
except:
    pass

non_eq_vel_errors = np.linalg.norm(non_eq_vel_data - non_eq_ref_vel_data, axis=-1)
non_eq_mean_vel_errors = np.mean(non_eq_vel_errors, axis=1)
non_eq_std_vel_errors = np.std(non_eq_vel_errors, axis=1)

p_eq_vel_errors = np.linalg.norm(p_eq_vel_data - p_eq_ref_vel_data, axis=-1)
p_eq_mean_vel_errors = np.mean(p_eq_vel_errors, axis=1)
p_eq_std_vel_errors = np.std(p_eq_vel_errors, axis=1)

v_eq_vel_errors = np.linalg.norm(v_eq_vel_data - v_eq_ref_vel_data, axis=-1)
v_eq_mean_vel_errors = np.mean(v_eq_vel_errors, axis=1)
v_eq_std_vel_errors = np.std(v_eq_vel_errors, axis=1)

pv_eq_vel_errors = np.linalg.norm(pv_eq_vel_data - pv_eq_ref_vel_data, axis=-1)
pv_eq_mean_vel_errors = np.mean(pv_eq_vel_errors, axis=1)
pv_eq_std_vel_errors = np.std(pv_eq_vel_errors, axis=1)

try:
    pva_eq_vel_errors = np.linalg.norm(pva_eq_vel_data - pva_eq_ref_vel_data, axis=-1)
    pva_eq_mean_vel_errors = np.mean(pva_eq_vel_errors, axis=1)
    pva_eq_std_vel_errors = np.std(pva_eq_vel_errors, axis=1)
except:
    pass

# TODO: Find rollout_end from eval_policy.py
rollout_end = 2000

plt.figure()
plt.plot(np.arange(rollout_end), non_eq_mean_pos_errors[:rollout_end], label="Non-Eq Mean Error")
plt.fill_between(np.arange(rollout_end), non_eq_mean_pos_errors[:rollout_end] - non_eq_std_pos_errors[:rollout_end], non_eq_mean_pos_errors[:rollout_end] + non_eq_std_pos_errors[:rollout_end], alpha=0.5)
plt.plot(np.arange(rollout_end), p_eq_mean_pos_errors[:rollout_end], label="P-Eq Mean Error")
plt.fill_between(np.arange(rollout_end), p_eq_mean_pos_errors[:rollout_end] - p_eq_std_pos_errors[:rollout_end], p_eq_mean_pos_errors[:rollout_end] + p_eq_std_pos_errors[:rollout_end], alpha=0.5)
plt.plot(np.arange(rollout_end), v_eq_mean_pos_errors[:rollout_end], label="V-Eq Mean Error")
plt.fill_between(np.arange(rollout_end), v_eq_mean_pos_errors[:rollout_end] - v_eq_std_pos_errors[:rollout_end], v_eq_mean_pos_errors[:rollout_end] + v_eq_std_pos_errors[:rollout_end], alpha=0.5)
plt.plot(np.arange(rollout_end), pv_eq_mean_pos_errors[:rollout_end], label="PV-Eq Mean Error")
plt.fill_between(np.arange(rollout_end), pv_eq_mean_pos_errors[:rollout_end] - pv_eq_std_pos_errors[:rollout_end], pv_eq_mean_pos_errors[:rollout_end] + pv_eq_std_pos_errors[:rollout_end], alpha=0.5)
try:
    plt.plot(np.arange(rollout_end), pva_eq_mean_pos_errors[:rollout_end], label="PVA-Eq Mean Error")
    plt.fill_between(np.arange(rollout_end), pva_eq_mean_pos_errors[:rollout_end] - pva_eq_std_pos_errors[:rollout_end], pva_eq_mean_pos_errors[:rollout_end] + pva_eq_std_pos_errors[:rollout_end], alpha=0.5)
except:
    pass
    
plt.xlabel("Timesteps")
plt.ylabel("Error")
plt.grid("True")
plt.legend()
plt.title(f"Mean Error Between Particle Position and Reference Position")
plt.tight_layout()
plt.savefig(data_dir + env_name +"_compare_mean_pos_error.png", dpi=1000)
# plt.show()
plt.close()


plt.figure()
plt.plot(np.arange(rollout_end), non_eq_mean_vel_errors[:rollout_end], label="Non-Eq Mean Error")
plt.fill_between(np.arange(rollout_end), non_eq_mean_vel_errors[:rollout_end] - non_eq_std_vel_errors[:rollout_end], non_eq_mean_vel_errors[:rollout_end] + non_eq_std_vel_errors[:rollout_end], alpha=0.5)
plt.plot(np.arange(rollout_end), p_eq_mean_vel_errors[:rollout_end], label="P-Eq Mean Error")
plt.fill_between(np.arange(rollout_end), p_eq_mean_vel_errors[:rollout_end] - p_eq_std_vel_errors[:rollout_end], p_eq_mean_vel_errors[:rollout_end] + p_eq_std_vel_errors[:rollout_end], alpha=0.5)
plt.plot(np.arange(rollout_end), v_eq_mean_vel_errors[:rollout_end], label="V-Eq Mean Error")
plt.fill_between(np.arange(rollout_end), v_eq_mean_vel_errors[:rollout_end] - v_eq_std_vel_errors[:rollout_end], v_eq_mean_vel_errors[:rollout_end] + v_eq_std_vel_errors[:rollout_end], alpha=0.5)
plt.plot(np.arange(rollout_end), pv_eq_mean_vel_errors[:rollout_end], label="PV-Eq Mean Error")
plt.fill_between(np.arange(rollout_end), pv_eq_mean_vel_errors[:rollout_end] - pv_eq_std_vel_errors[:rollout_end], pv_eq_mean_vel_errors[:rollout_end] + pv_eq_std_vel_errors[:rollout_end], alpha=0.5)
try:
    plt.plot(np.arange(rollout_end), pva_eq_mean_vel_errors[:rollout_end], label="PVA-Eq Mean Error")
    plt.fill_between(np.arange(rollout_end), pva_eq_mean_vel_errors[:rollout_end] - pva_eq_std_vel_errors[:rollout_end], pva_eq_mean_vel_errors[:rollout_end] + pva_eq_std_vel_errors[:rollout_end], alpha=0.5)
except:
    pass
    
plt.xlabel("Timesteps")
plt.ylabel("Error")
plt.grid("True")
plt.legend()
plt.title(f"Mean Error Between Particle Velocity and Reference Velocity")
plt.tight_layout()
plt.savefig(data_dir + env_name + "_compare_mean_vel_error.png", dpi=1000)
# plt.show()
plt.close()
