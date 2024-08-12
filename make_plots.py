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

data_dir = "./checkpoints/current_baselines/"

non_eq_data_path = data_dir + env_name + "/training_data.npz"
p_eq_data_path = data_dir + env_name + "_p_equivariant/training_data.npz"
v_eq_data_path = data_dir + env_name + "_v_equivariant/training_data.npz"
pv_eq_data_path = data_dir + env_name + "_pv_equivariant/training_data.npz"
try:
    pva_eq_data_path = data_dir + env_name + "_pva_equivariant/training_data.npz"
except:
    pva_eq_data_path = None
# eq_data_path = "./checkpoints/constant_velocity_eq_50M_1/training_data.npz"
# non_eq_data_path = "./checkpoints/constant_velocity_no_eq_50M/training_data.npz"

p_eq_data = np.load(p_eq_data_path)
v_eq_data = np.load(v_eq_data_path)
pv_eq_data = np.load(pv_eq_data_path)
non_eq_data = np.load(non_eq_data_path)
if pva_eq_data_path is not None:
    pva_eq_data = np.load(pv_eq_data_path)


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