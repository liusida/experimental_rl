# experimental_rl

Plan:

VAEs + RNNs + RL

# Useful commands

```bash
# Install dependencies:
pip install -r requirements.txt
# Install current project as a package:
pip install -e .
# Run unittest:
pytest
```

# Issues
Tensorboard 2.4.1 is using `np.bool` and `np.object`, which are deprecated. 
To avoid warning, replace those by `bool` and `object`.

# DeepGreen commands

```
# allocate a node
srun -p dg-jup --gres=gpu:1 --pty bash
# load newest software pack
module load spack/spack-0.15.4
. $SPACK_ROOT/share/spack/setup-env.sh
# run
python run.py
```