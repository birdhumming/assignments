# P4.0 — Scaling rules from alignment assumptions

## Setup

Width \(n\), reference width \(n_0\), \(m = n/n_0\). Every matrix is used as \(n^{-a} W\) in the forward pass,
initialized with i.i.d. entries of variance \(n^{-2b}\), and updated by \(\Delta W = -\eta\, n^{-c} U\) with
\(\|U\|_{\mathrm{RMS}} = \Theta(1)\). Fix \(a = 0\) for hidden and input matrices and \(c = 0\) for the readout.

Two alignment ratios (RMS norms throughout):

* update vs. pre-update input: \(R(U, x_0) = \|U x_0\| / (\|U\|\,\|x_0\|) = \Theta(n^{\alpha})\),
  \(\alpha = 1\) if the update direction is aligned with the input (every one of the \(n\) terms in the sum adds
  coherently) and \(\alpha = 1/2\) if not (random-walk sum of \(n\) terms);
* initial readout vs. feature change: \(S(V_0, \Delta x) = \Theta(n^{\omega})\), \(\omega = 1\) aligned, \(1/2\) not.

For the interaction term assume \(R(U, \Delta x) = O(n^{\alpha})\).

## (a) Constraints

**Hidden matrix** (\(a = 0\), fan-in \(k = n\)).

1. Order-one features at initialization: \(\|W x_0\| \sim \sqrt{n}\, n^{-b} = \Theta(1)\Rightarrow b = 1/2\) (variance \(1/k\)).
   Independent of alignment.
2. Order-one direct update: \(\|\Delta W x_0\| = \eta\, n^{-c} R(U, x_0) = \eta\, n^{\alpha - c} = \Theta(1) \Rightarrow c = \alpha\).
3. Interaction: \(\|\Delta W \Delta x\| = \eta\, n^{-c} R(U,\Delta x) \|\Delta x\| = O(n^{\alpha - c}) = O(1)\). Bounded automatically once \(c = \alpha\).

**Readout** (\(c = 0\), transposed convention \(V_0 \in \mathbb{R}^{n\times q}\), response \(n^{-a} V^\top x\)).

1. Order-one direct update: \(n^{-a} \|\Delta V^\top x_0\| = \eta\, n^{-a} R(U, x_0) = \eta\, n^{\alpha - a} \Rightarrow a = \alpha\).
2. Order-one response of the *initial* readout to the feature change:
   \(n^{-a}\|V_0^\top \Delta x\| = n^{-a} S(V_0,\Delta x)\, n^{-b} = n^{\omega - a - b} = \Theta(1) \Rightarrow b = \omega - a = \omega - \alpha\).
3. Interaction: \(n^{-a}\|\Delta V^\top \Delta x\| = \eta\, n^{\alpha - a} = O(1)\). Bounded.

**Input / embedding matrix** (fixed fan-in: a token hits one column, so no sum over \(n\)).

1. Order-one features: each column entry is the feature, so \(b = 0\) (variance 1).
2. Order-one update: \(\|\Delta W e_j\| = \eta\, n^{-c} \|U_{:,j}\| = \eta\, n^{-c} \Rightarrow c = 0\). Alignment plays no role.

| Alignment \((\alpha, \omega)\) | Hidden \((a,b,c)\) | Readout \((a,b,c)\) | Input \((a,b,c)\) |
|---|---|---|---|
| update aligned, readout aligned \((1, 1)\) | \((0, 1/2, 1)\) | \((1, 0, 0)\) | \((0,0,0)\) |
| update aligned, readout not \((1, 1/2)\) | \((0, 1/2, 1)\) | \((1, -1/2, 0)\) | \((0,0,0)\) |
| update not aligned, readout aligned \((1/2, 1)\) | \((0, 1/2, 1/2)\) | \((1/2, 1/2, 0)\) | \((0,0,0)\) |
| neither aligned \((1/2, 1/2)\) | \((0, 1/2, 1/2)\) | \((1/2, 0, 0)\) | \((0,0,0)\) |

## (b) Rules in terms of \(m = n/n_0\)

Matching the table at \(n_0\): forward multiplier \(m^{-a}\), initialization variance \(\sigma_0^2\, m^{-2b}\)
(\(\sigma_0^2\) the reference-width value), learning rate \(\eta\, m^{-c}\).

| Layer | multiplier | init variance | LR |
|---|---|---|---|
| hidden | 1 | \(1/k\) | \(\eta\, m^{-\alpha}\) |
| readout | \(m^{-\alpha}\) | \(n_0^{-1} m^{-2(\omega - \alpha)}\) | \(\eta\) |
| embedding | 1 | 1 | \(\eta\) |

Effect of each assumption:

* Update alignment \(\alpha\) sets how fast hidden LRs must shrink (\(1/m\) if aligned, \(1/\sqrt m\) if not) and how
  strongly the readout must be damped (\(1/m\) vs \(1/\sqrt m\)). Adam's updates are sign-like and strongly correlated
  with the input, so \(\alpha = 1\) is the right working assumption for hidden matrices.
* Readout alignment \(\omega\) only changes the readout's initialization: with \(\omega = \alpha\) the variance is
  width-independent (\(1/n_0\)); with \(\omega < \alpha\) it would have to *grow* with width (\(m^{+1}\)), which is
  the signal that "aligned updates but unaligned readout" is not a consistent regime.
* The interaction term \(n^{-a}\Delta V^\top\Delta x\) and \(\Delta W \Delta x\) are both \(O(n^{\alpha - c})\) or
  \(O(n^{\alpha - a})\), so they stay bounded in every row once the direct-update constraint holds.

**Which row is µP.** \((\alpha,\omega) = (1,1)\): hidden variance \(1/k\) and LR \(\eta/m\); readout variance
\(1/n_0\), multiplier \(1/m\), LR \(\eta\); embedding variance 1 and LR \(\eta\). This is exactly the P4.1 µP column
(and, at \(n_0 = 512\), the P4.2 µP column with \(G/n_0\), \(G/\sqrt{n_0}\), \(1/m\), \(\eta/m\)).

**Where Kaiming sits.** Kaiming keeps readout \((a,b,c) = (0, 1/2, 0)\) and hidden \(c = 0\). Its readout satisfies
\(a + b = 1/2 = \omega\) only for an *unaligned* readout, and its hidden/readout direct updates grow like
\(\eta\, n^{\alpha}\): with aligned Adam updates the first step moves features by \(\Theta(n)\) at fixed \(\eta\), so
the tuned base LR must fall roughly like \(1/n\) as width grows. That is the prediction P4.1 tests.
