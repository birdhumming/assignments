# Assignment 2: Hyperparameter Scaling

Read [the student handout](assignment2.pdf) and follow the
[A2 runtime guide](../../experiments/a2/README.md) for setup and experiments.

The handout includes an example-question answer key. The [data guide](data/README.md)
describes the 78 supplied source runs and width-512 diagnostics.

The assignment covers:

1. Learning-rate scaling with training horizon.
2. Joint learning-rate and weight-decay scaling.
3. Batch size: noisy-quadratic simulations and language-model experiments.
4. Width and depth: five-step Transformer tests and longer training.

For Problem 4.1, complete the two policy functions in
`experiments/a2/p31_student.py` using your derivations.

## Build the handout

From the repository root:

```sh
make -B -C worksheets/hparam_invariants
```

Requires Tectonic or latexmk. The output is `assignment2.pdf`.
