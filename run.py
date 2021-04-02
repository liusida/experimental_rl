import argparse
import erl


parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Show the monitor when local.")
parser.add_argument("--total_timesteps", type=float, default=1e8, help="Total training time, measured in steps.")
parser.add_argument("--num_venvs", type=int, default=16, help="Number of vectorized environments.")
parser.add_argument("--extractor", type=str, default="TwoMlpExtractor", help="Specify different extractor. The main treatment of the experiment.")
args = parser.parse_args()

erl.run_current_exp(args)