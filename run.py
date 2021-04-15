from collections import defaultdict
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
parser.add_argument("--exp_group", type=str, default="baseline", help="could be (1) baseline, (2) mlps, (3) rnns, (4) mlprnn")
parser.add_argument("--extractor", type=str, default="MultiMlpExtractor:m=4", help="Specify different extractor. The main treatment of the experiment.")
parser.add_argument("--env_id", type=str, default="HopperBulletEnv-v0", help="Specify different environment. The secondary treatment of the experiment.")

parser.add_argument("--rollout-n-steps", type=int, default=2048, help="n_steps for CustomizedPPO.__init__()")
parser.add_argument("--eval_freq", type=int, default=10000, help="eval_freq for CustomizedEvalCallback.__init__()")
args = parser.parse_args()

args.commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])

erl.run_current_exp(args)