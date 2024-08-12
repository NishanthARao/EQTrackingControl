echo "==============================================================="
echo "Running Training Experiments...."
echo "==============================================================="
python3 train_policy.py --env-name position --exp-name test_position --add-desc "Non-Eq version, with termination on error and bad, good reward"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name position --exp-name test_position_p_equivariant --equivariant 1 --add-desc "P-Eq version, with termination on error and bad, good reward"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name position --exp-name test_position_v_equivariant --equivariant 2 --add-desc "V-Eq version, with termination on error and bad, good reward"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name position --exp-name test_position_pv_equivariant --equivariant 3 --add-desc "PV-Eq version, with termination on error and bad, good reward"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name constant_velocity --exp-name test_constant_velocity --add-desc " Non-Eq version, with termination on error and bad, good reward, constant velocity sampled randomly"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name constant_velocity --exp-name test_constant_velocity_p_equivariant --equivariant 1 --add-desc "P-Eq version, with termination on error and bad, good reward, constant velocity sampled randomly"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name constant_velocity --exp-name test_constant_velocity_v_equivariant --equivariant 2 --add-desc "V-Eq version, with termination on error and bad, good reward, constant velocity sampled randomly"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name constant_velocity --exp-name test_constant_velocity_pv_equivariant --equivariant 3 --add-desc "PV-Eq version, with termination on error and bad, good reward, constant velocity sampled randomly"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_position --exp-name test_random_walk_position --add-desc "Non-Eq version, with termination on error and bad, good reward and random velocity sampled each time instant"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_position --exp-name test_random_walk_position_p_equivariant --equivariant 1 --add-desc "P-Eq version, with termination on error and bad, good reward and random velocity sampled each time instant"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_position --exp-name test_random_walk_position_v_equivariant --equivariant 2 --add-desc "V-Eq version, with termination on error and bad, good reward and random velocity sampled each time instant"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_position --exp-name test_random_walk_position_pv_equivariant --equivariant 3 --add-desc "PV-Eq version, with termination on error and bad, good reward and random velocity sampled each time instant"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_walk_velocity --add-desc "Non-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_walk_velocity_p_equivariant --equivariant 1 --add-desc "P-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_walk_velocity_v_equivariant --equivariant 2 --add-desc "V-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_walk_velocity_pv_equivariant --equivariant 3 --add-desc "PV-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_walk_velocity_pva_equivariant --equivariant 4 --out-activation hard_tanh_scaled --add-desc "PVA-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_lissajous --equivariant 0 --add-desc "Non-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function, folder made for Lissajous testing"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_lissajous_p_equivariant --equivariant 1 --add-desc "P-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function, folder made for Lissajous P-Eq testing"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_lissajous_v_equivariant --equivariant 2 --add-desc "V-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function, folder made for Lissajous V-Eq testing"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_lissajous_pv_equivariant --equivariant 3 --add-desc "PV-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function, folder made for Lissajous PV-Eq testing"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_lissajous_pva_equivariant --equivariant 4 --out-activation hard_tanh_scaled --add-desc "PVA-Eq version, with termination on error and bad, good reward and random accel sampled each time instant, and desired action in the reward function, folder made for Lissajous PVA-Eq testing"
sleep 2
echo "==============================================================="
#use --make-animation to also generate animation (albeit slow)

echo "==============================================================="
echo "Running Evaluation Experiments...."
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_position/model_final/ --env-name position
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_position_p_equivariant/model_final/ --env-name position --equivariant 1
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_position_v_equivariant/model_final/ --env-name position --equivariant 2
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_position_pv_equivariant/model_final/ --env-name position --equivariant 3
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_constant_velocity/model_final/ --env-name constant_velocity
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_constant_velocity_p_equivariant/model_final/ --env-name constant_velocity --equivariant 1
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_constant_velocity_v_equivariant/model_final/ --env-name constant_velocity --equivariant 2
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_constant_velocity_pv_equivariant/model_final/ --env-name constant_velocity --equivariant 3
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_position/model_final/ --env-name random_walk_position
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_position_p_equivariant/model_final/ --env-name random_walk_position --equivariant 1
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_position_v_equivariant/model_final/ --env-name random_walk_position --equivariant 2
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_position_pv_equivariant/model_final/ --env-name random_walk_position --equivariant 3
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_velocity/model_final/ --env-name random_walk_velocity
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_velocity_p_equivariant/model_final/ --env-name random_walk_velocity --equivariant 1
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_velocity_v_equivariant/model_final/ --env-name random_walk_velocity --equivariant 2
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_velocity_pv_equivariant/model_final/ --env-name random_walk_velocity --equivariant 3
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_lissajous/model_final/ --env-name random_lissajous --equivariant 0
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_lissajous_p_equivariant/model_final/ --env-name random_lissajous --equivariant 1
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_lissajous_v_equivariant/model_final/ --env-name random_lissajous --equivariant 2
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_lissajous_pv_equivariant/model_final/ --env-name random_lissajous --equivariant 3
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_lissajous_pva_equivariant/model_final/ --env-name random_lissajous --equivariant 4
sleep 2
echo "==============================================================="
python3 make_plots.py test_position
sleep 2
echo "==============================================================="
python3 make_plots.py test_constant_velocity
sleep 2
echo "==============================================================="
python3 make_plots.py test_random_walk_position
sleep 2
echo "==============================================================="
python3 make_plots.py test_random_walk_velocity
sleep 2
echo "==============================================================="
python3 make_plots.py test_random_lissajous
sleep 2
echo "==============================================================="
echo "Huh... Done!"
