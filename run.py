import argparse
import erl


parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Show the monitor when local")
args = parser.parse_args()

erl.run1(render=args.render)