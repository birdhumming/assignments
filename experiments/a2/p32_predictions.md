# P3.2 predictions, recorded before any B=128/256 run (2026-10-02)

Sources: our B=8/16/32 runs (a2-p32a) plus the supplied B=64 P1(a)/P2(a) curves, all 614.4M tokens, beta1 .9, beta2 .95.

Hypothesis i (WD fixed at .1, scale LR): lr* = 0.000369 * B^0.559 (fit on B=8..64)
  -> B=128: 0.00556, B=256: 0.00819.  Linear scaling (NQM low-noise, B^1) would say 0.0064 / 0.0128.
  Sweeps: B=128 {pred, .003, .006, .012}; B=256 {pred, .006, .012, .024}.

Hypothesis ii (LR fixed at .0015, scale WD): wd* = 0.0129 * B^0.801 on interior points B=8,16
  (B=32 and 64 best at the grid edge .2 / .4 -> lower bounds). -> B=128: 0.63, B=256: 1.10.
  Linear wd* ∝ B anchored at B=16: 0.95 / 1.91.
  Sweeps: B=128 {pred, .4, .8, 1.6}; B=256 {pred, .8, 1.6, 3.2}.

Prediction: hypothesis i should win on loss at both targets (LR scaling moved the B=8->64 loss floor
from 2.934 to 2.919; the fixed-LR/WD-scaled curves are flatter and sit ~0.003 higher), but the gap
narrows at B=256 because the LR exponent is already sub-linear (0.56) and the loss-vs-batch curve is turning up at B=64.

Momentum (c), trimmed: beta1 in {0, .5, .98} at each batch's best measured (LR, WD): B=8 -> (.0015, .05); B=256 -> decided after targets.
Prediction: beta1=0 hurts little at B=8 (small-batch noise averages anyway) and hurts more at B=256 (NQM: momentum helped at B=256 only once LR was retuned down).
