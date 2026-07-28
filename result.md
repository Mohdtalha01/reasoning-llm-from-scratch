numeric=-1.468218 analytic=-1.468218 rel_err=1.85e-09
numeric= 0.119208 analytic= 0.119208 rel_err=4.16e-09
numeric= 0.313739 analytic= 0.313739 rel_err=6.26e-09
numeric=-2.162860 analytic=-2.162860 rel_err=4.37e-09
numeric=-0.095801 analytic=-0.095801 rel_err=5.48e-10
numeric=-0.517472 analytic=-0.517472 rel_err=8.32e-10
numeric= 0.008621 analytic= 0.008621 rel_err=2.43e-10
numeric= 0.108691 analytic= 0.108691 rel_err=8.26e-11
numeric=-0.027087 analytic=-0.027087 rel_err=1.66e-10
numeric= 0.119624 analytic= 0.119624 rel_err=9.94e-12
numeric= 0.003611 analytic= 0.003611 rel_err=2.90e-09
numeric=-0.096004 analytic=-0.096004 rel_err=4.65e-11
numeric= 0.020199 analytic= 0.020199 rel_err=7.76e-10
numeric= 0.009666 analytic= 0.009666 rel_err=1.21e-09
numeric=-0.004310 analytic=-0.004310 rel_err=8.18e-10
numeric= 0.000676 analytic= 0.000676 rel_err=2.47e-08
numeric= 0.007890 analytic= 0.007890 rel_err=1.68e-09
numeric=-0.006941 analytic=-0.006941 rel_err=3.78e-10
numeric= 0.003138 analytic= 0.003138 rel_err=7.54e-10
numeric=-0.037605 analytic=-0.037605 rel_err=2.17e-10
numeric=-0.008635 analytic=-0.008635 rel_err=7.85e-10
numeric= 0.000000 analytic= 0.000000 rel_err=2.39e-10
numeric= 0.000000 analytic=-0.000000 rel_err=2.60e-10
numeric= 0.000000 analytic= 0.000000 rel_err=1.73e-10
numeric= 0.177694 analytic= 0.177694 rel_err=1.89e-11
numeric= 0.247147 analytic= 0.247147 rel_err=3.58e-11
numeric= 0.033196 analytic= 0.033196 rel_err=1.50e-10
numeric= 0.166061 analytic= 0.166061 rel_err=6.16e-11
numeric= 0.038898 analytic= 0.038898 rel_err=2.11e-10
numeric= 0.025627 analytic= 0.025627 rel_err=2.87e-10
numeric= 0.065188 analytic= 0.065188 rel_err=2.30e-10
numeric=-0.164076 analytic=-0.164076 rel_err=8.24e-11
numeric= 0.086770 analytic= 0.086770 rel_err=3.56e-11
numeric=-0.307869 analytic=-0.307869 rel_err=7.75e-11
numeric= 0.201246 analytic= 0.201246 rel_err=1.66e-10
numeric=-0.021682 analytic=-0.021682 rel_err=4.84e-10
numeric= 0.050116 analytic= 0.050116 rel_err=3.97e-10
numeric=-0.023977 analytic=-0.023977 rel_err=7.72e-10
numeric= 0.018906 analytic= 0.018906 rel_err=3.33e-10
numeric= 0.030339 analytic= 0.030339 rel_err=5.34e-11
numeric=-0.017985 analytic=-0.017985 rel_err=4.27e-10
numeric= 0.019395 analytic= 0.019395 rel_err=2.64e-11
numeric=-0.060393 analytic=-0.060393 rel_err=1.72e-10
numeric= 0.013264 analytic= 0.013264 rel_err=2.79e-10
numeric= 0.000000 analytic= 0.000000 rel_err=0.00e+00
numeric=-0.014632 analytic=-0.014632 rel_err=6.31e-10
numeric=-0.013678 analytic=-0.013678 rel_err=5.30e-10
numeric= 0.000000 analytic= 0.000000 rel_err=0.00e+00
numeric=-0.014365 analytic=-0.014365 rel_err=7.36e-11
numeric= 0.000000 analytic= 0.000000 rel_err=0.00e+00
numeric=-0.014135 analytic=-0.014135 rel_err=1.07e-09
numeric=-0.061910 analytic=-0.061910 rel_err=8.67e-12
numeric=-0.009757 analytic=-0.009757 rel_err=2.58e-10
numeric= 0.074097 analytic= 0.074097 rel_err=1.05e-11
numeric= 0.006022 analytic= 0.006022 rel_err=8.69e-10
numeric=-0.003476 analytic=-0.003476 rel_err=3.81e-09
numeric= 0.129322 analytic= 0.129322 rel_err=5.82e-11
numeric= 0.071944 analytic= 0.071944 rel_err=1.39e-11
numeric= 0.077849 analytic= 0.077849 rel_err=2.48e-11
numeric= 0.009292 analytic= 0.009292 rel_err=1.24e-09
numeric=-0.038858 analytic=-0.038858 rel_err=1.81e-10
numeric= 0.015037 analytic= 0.015037 rel_err=4.12e-10
numeric=-0.003773 analytic=-0.003773 rel_err=3.55e-10
numeric=-0.152452 analytic=-0.152452 rel_err=1.25e-11
numeric=-0.009975 analytic=-0.009975 rel_err=6.95e-10
numeric=-0.014622 analytic=-0.014622 rel_err=3.07e-10

reasoning_llm arjun$ python3 train.py --steps 5000 --eval_every 250 \
>     --block_size 48 --batch_size 16 --d_model 48 --n_layers 2 --n_heads 4 --d_ff 192 \
>     --out checkpoint.npz
vocab_size=49  params=63,697
step     1 | train_loss 3.958 | val_loss 3.546 | 0.2s elapsed
step   250 | train_loss 0.362 | val_loss 0.372 | 27.4s elapsed
step   500 | train_loss 0.304 | val_loss 0.271 | 52.5s elapsed
step   750 | train_loss 0.346 | val_loss 0.371 | 76.6s elapsed
step  1000 | train_loss 0.363 | val_loss 0.383 | 102.6s elapsed
step  1250 | train_loss 0.277 | val_loss 0.295 | 127.8s elapsed
step  1500 | train_loss 0.252 | val_loss 0.396 | 151.2s elapsed
step  1750 | train_loss 0.280 | val_loss 0.411 | 182.0s elapsed
step  2000 | train_loss 0.435 | val_loss 0.279 | 211.4s elapsed
step  2250 | train_loss 0.336 | val_loss 0.292 | 238.0s elapsed
step  2500 | train_loss 0.261 | val_loss 0.324 | 268.2s elapsed
step  2750 | train_loss 0.264 | val_loss 0.297 | 294.0s elapsed
step  3000 | train_loss 0.363 | val_loss 0.391 | 318.4s elapsed
step  3250 | train_loss 0.319 | val_loss 0.401 | 345.8s elapsed
step  3500 | train_loss 0.311 | val_loss 0.328 | 369.1s elapsed
step  3750 | train_loss 0.284 | val_loss 0.302 | 394.4s elapsed
step  4000 | train_loss 0.321 | val_loss 0.279 | 418.3s elapsed
step  4250 | train_loss 0.295 | val_loss 0.327 | 441.5s elapsed
step  4500 | train_loss 0.272 | val_loss 0.293 | 465.2s elapsed
step  4750 | train_loss 0.242 | val_loss 0.310 | 491.0s elapsed
step  5000 | train_loss 0.328 | val_loss 0.334 | 518.9s elapsed
Saved checkpoint.npz
python3 train.py --steps 2000 --resume checkpoint.npz --out checkpoint.npz \
>     --block_size 48 --batch_size 16 --d_model 48 --n_layers 2 --n_heads 4 --d_ff 192
Resumed weights from checkpoint.npz
vocab_size=49  params=63,697
step     1 | train_loss 0.231 | val_loss 0.289 | 0.1s elapsed
step   200 | train_loss 0.324 | val_loss 0.255 | 18.2s elapsed
step   400 | train_loss 0.282 | val_loss 0.271 | 38.2s elapsed
step   600 | train_loss 0.314 | val_loss 0.274 | 56.8s elapsed
step   800 | train_loss 0.386 | val_loss 0.290 | 74.4s elapsed
step  1000 | train_loss 0.232 | val_loss 0.323 | 98.3s elapsed
step  1200 | train_loss 0.260 | val_loss 0.268 | 119.2s elapsed
step  1400 | train_loss 0.264 | val_loss 0.340 | 139.8s elapsed
step  1600 | train_loss 0.302 | val_loss 0.302 | 159.6s elapsed
step  1800 | train_loss 0.360 | val_loss 0.306 | 179.9s elapsed
step  2000 | train_loss 0.247 | val_loss 0.306 | 200.7s elapsed
Saved checkpoint.npz
python3 train.py --steps 5000
vocab_size=49  params=110,513
step     1 | train_loss 4.085 | val_loss 3.405 | 0.7s elapsed
step   200 | train_loss 0.351 | val_loss 0.347 | 90.8s elapsed
step   400 | train_loss 0.318 | val_loss 0.311 | 160.5s elapsed
step   600 | train_loss 0.305 | val_loss 0.249 | 214.0s elapsed
step   800 | train_loss 0.300 | val_loss 0.336 | 281.8s elapsed
step  1000 | train_loss 0.280 | val_loss 0.306 | 351.1s elapsed
step  1200 | train_loss 0.319 | val_loss 0.271 | 416.2s elapsed
step  1400 | train_loss 0.312 | val_loss 0.328 | 484.2s elapsed
step  1600 | train_loss 0.275 | val_loss 0.273 | 548.2s elapsed
step  1800 | train_loss 0.304 | val_loss 0.235 | 614.9s elapsed
step  2000 | train_loss 0.260 | val_loss 0.301 | 690.2s elapsed
step  2200 | train_loss 0.280 | val_loss 0.237 | 760.6s elapsed
step  2400 | train_loss 0.310 | val_loss 0.257 | 828.5s elapsed
step  2600 | train_loss 0.240 | val_loss 0.281 | 896.3s elapsed
step  2800 | train_loss 0.261 | val_loss 0.275 | 966.4s elapsed
step  3000 | train_loss 0.293 | val_loss 0.308 | 1036.6s elapsed
step  3200 | train_loss 0.283 | val_loss 0.254 | 1117.8s elapsed
step  3400 | train_loss 0.297 | val_loss 0.278 | 1205.9s elapsed
step  3600 | train_loss 0.290 | val_loss 0.279 | 1306.9s elapsed
step  3800 | train_loss 0.255 | val_loss 0.260 | 1413.6s elapsed
step  4000 | train_loss 0.250 | val_loss 0.281 | 1515.4s elapsed
step  4200 | train_loss 0.276 | val_loss 0.312 | 1612.2s elapsed
step  4400 | train_loss 0.263 | val_loss 0.265 | 1682.4s elapsed
step  4600 | train_loss 0.276 | val_loss 0.259 | 1743.5s elapsed
step  4800 | train_loss 0.291 | val_loss 0.238 | 1812.7s elapsed
step  5000 | train_loss 0.285 | val_loss 0.254 | 1877.7s elapsed
Saved checkpoint.npz
python3 evaluate.py


=== Evaluating: TinyGPT (ours, from-scratch transformer) ===
 
val perplexity        : 1.300
benchmark accuracy     : 56.2%
  - chain_reasoning   :  12.0%
  - comparison        :  92.0%
  - edge_arithmetic   :  66.7%
  - counting          :  88.0%
  - arithmetic        :  32.0%
  - edge_counting     : 100.0%
  - edge_comparison   :   0.0%
avg latency/answer     : 44.0 ms

=== Evaluating: N-gram baseline (order-5 Markov chain) ===
val perplexity        : 1.465
benchmark accuracy     : 0.0%
  - chain_reasoning   :   0.0%
  - comparison        :   0.0%
  - edge_arithmetic   :   0.0%
  - counting          :   0.0%
  - arithmetic        :   0.0%
  - edge_counting     :   0.0%
  - edge_comparison   :   0.0%
avg latency/answer     : 0.2 ms

========================================================================
metric                                 tinygpt               ngram
------------------------------------------------------------------------
val perplexity                           1.300               1.465
benchmark accuracy                       56.2%                0.0%
avg latency (ms)                          44.0                 0.2
========================================================================
