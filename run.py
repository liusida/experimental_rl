import argparse
import erl


parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Show the monitor when local.")
parser.add_argument("--total_timesteps", type=int, default=1e8, help="Total training time, measured in steps.")
args = parser.parse_args()

erl.run1(render=args.render)