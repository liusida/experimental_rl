import subprocess
import argparse
import erl


parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Show the monitor when local.")
parser.add_argument("--total_timesteps", type=float, default=1e8, help="Total training time, measured in steps.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of vectorized environments.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cuda", action="store_true", help="use cuda if possible.")

parser.add_argument("--exp_name", type=str, default="Default", help="Name of the experiment. So we can group them on WandB.")

parser.add_argument("--flatten", action="store_true", help="Whether include flattened input as features")
parser.add_argument("--num_rnns", type=int, default=0, help="Number of RNNs")
parser.add_argument("--num_mlps", type=int, default=0, help="Number of MLPs")

parser.add_argument("--env_id", type=str, default="HopperBulletEnv-v0", help="Specify different environment. The secondary treatment of the experiment.")

parser.add_argument("--rollout_n_steps", type=int, default=2048, help="n_steps for CustomizedPPO.__init__()")
parser.add_argument("--eval_freq", type=int, default=10000, help="eval_freq for CustomizedEvalCallback.__init__()")

parser.add_argument("--implementation_check", action="store_true", help="use default implementation, check with flatten version, to make sure the implementation of flatten version is correct.")
parser.add_argument("--vec_normalize", action="store_true", help="use sb3 VecNormalization to improve performance.")
parser.add_argument("--rnn_move_window_step", type=int, default=16, help="set to 1 for data efficiency, set to rollout_n_steps to get faster wall time.")
parser.add_argument("--rnn_sequence_length", type=int, default=16, help="Length of a sequence for RNN to learn. (We don't BPTT for the whole episode, no need)")
parser.add_argument("--rnn_layer_size", type=int, default=16, help="hidden layer size for hx and cx.")
parser.add_argument("--sde", action="store_true", help="use gSDE exploration.")
parser.add_argument("--n_epochs", type=int, default=1, help="Since RNN has much more chance to learn, there's no need to use multiple epochs to learn from the same data. (I guess)")

parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--exp_desc", type=str, default="")

parser.add_argument("--cnn", action="store_true", help="use CNN extractor")
args = parser.parse_args()

assert(args.rollout_n_steps%args.rnn_move_window_step==0)

args.commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])

erl.run_current_exp(args)