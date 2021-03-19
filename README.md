# experimental_rl

Plan:

VAEs + RNNs + RL

# Useful command

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

