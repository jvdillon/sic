WORK IN PROGRESS

Preliminary Leaderboard

Step lines of the form:

```
Step <N>  Test <cell>% / <puzzle>% (halt: <cell>%/<puzzle>%) ... TrainAcc <cell>%/<puzzle>% (halt: <cell>%/<puzzle>%)
```
where first cell/puzzle is H16 and second cell/puzzle is at `sigmoid(q_halt)>0`.


```
x07b.log:Step  5000  Test 99.50% / 94.80% (halt: 99.49%/81.60%)  Train lm=0.6641±0.0026ϵ[0.6641,0.6914] n_halted=108.4480±12.5009ϵ[0.0000,123.0000]  TrainAcc 99.95%/97.10% (halt: 99.95%/96.60%)  (55.571s / 399.136s)
x08.log:Step 13000  Test 99.54% / 94.00% (halt: 99.51%/80.30%)  Train lm=0.6641±0.0027ϵ[0.6641,0.7148] n_halted=109.5080±9.1093ϵ[27.0000,122.0000]  TrainAcc 99.99%/98.20% (halt: 100.00%/98.50%)  (54.858s / 351.767s)
x07f.log:Step  9500  Test 99.54% / 93.60% (halt: 99.50%/82.30%)  Train lm=0.6641±0.0017ϵ[0.6641,0.6758] n_halted=109.3240±7.4623ϵ[59.0000,121.0000]  TrainAcc 99.98%/98.80% (halt: 99.98%/98.10%)  (54.258s / 178.159s)
x07.log.0:Step 16000  Test 99.47% / 93.10% (halt: 99.44%/81.20%)  Train lm=0.6641±0.0013ϵ[0.6641,0.6680] n_halted=110.4260±5.0354ϵ[82.0000,122.0000]  TrainAcc 99.99%/99.20% (halt: 99.98%/98.90%)  (55.402s / 180.375s)
x10.log:Step 12000  Test 99.50% / 93.40% (halt: 99.49%/83.20%)  Train lm=0.6641±0.0012ϵ[0.6641,0.6719] n_halted=108.4940±7.0220ϵ[62.0000,121.0000]  TrainAcc 99.98%/99.10% (halt: 99.98%/99.00%)  (55.416s / 288.770s)
```

[TRM repro](https://github.com/gaoxin492/TinyRecursiveModels#) uses 8 H(100?) series GPUs to get:
| Method | Params | Sudoku | Maze | ARC-1 (@2) | ARC-2 (@2) |
| --- | --- | --- | --- | --- | --- |
| TRM-Att | 7M | 77.71 | 78.70 | 41.00  | 3.33 |
| TRM-MLP | 5M | 84.80 | / | / | / |
| Time(h) | | 0.67 | 2 | 37 | 49 |

TODOs/Notes:
- still need to re-enable EMA to see smooth results. This will obviously bring down peak
- our puzzle acc is "valid solution"; unclear if gaoxin492 uses exact acc like TRM repo. This will cost ~5%.
