import argparse
import erl


parser = argparse.ArgumentParser()
parser.add_argument("--model_filename", type=str)
parser.add_argument("--vnorm_filename", type=str, default="")

args = parser.parse_args()
args.cuda = False
erl.test_current_exp(args)