# Optimizing julia Kalman filter implementation

This is based on the blogpost https://www.ronanarraes.com/2019/04/my-julia-workflow-readability-vs-performance/

1. [01.jl](./01.jl) - Basic implemetation
2. [01-timed.jl](01-timed.jl) - Basic implementation with timers for bottleneck analysis
3. [02.jl](02.jl) - Improvement over first version based on timed analysis.
4. [03.jl](03.jl) - Heavily optimized with all perallocations.
5. [03-timed.jl](03-timed.jl) - Timer outputs for [03.jl](./03.jl)

## Results
```
$ julia 01.jl
BenchmarkTools.Trial: 
  memory estimate:  2.34 GiB
  allocs estimate:  2520262
  --------------
  minimum time:     1.986 s (6.22% GC)
  median time:      2.056 s (6.80% GC)
  mean time:        2.066 s (6.56% GC)
  maximum time:     2.157 s (6.64% GC)
  --------------
  samples:          3
  evals/sample:     1
```
```
$ julia 01-timed.jl
 r[end] = 7.000894875771719
 ───────────────────────────────────────────────────────────────────────────
                                    Time                   Allocations      
                            ──────────────────────   ───────────────────────
      Tot / % measured:          8.48s / 86.6%           2.98GiB / 97.2%    

 Section            ncalls     time   %tot     avg     alloc   %tot      avg
 ───────────────────────────────────────────────────────────────────────────
 Main                    1    7.34s   100%   7.34s   2.90GiB  100%   2.90GiB
   Simulation            1    1.91s  26.1%   1.91s   2.34GiB  80.8%  2.34GiB
     loop                1    1.91s  26.1%   1.91s   2.34GiB  80.7%  2.34GiB
       K             60.0k    849ms  11.6%  14.2μs    604MiB  20.4%  10.3KiB
       Pu            60.0k    736ms  10.0%  12.3μs   1.29GiB  44.4%  22.5KiB
       Pp            60.0k    298ms  4.06%  4.97μs    472MiB  15.9%  8.06KiB
   Initialization        1    1.10s  15.0%   1.10s   81.0MiB  2.73%  81.0MiB
 ───────────────────────────────────────────────────────────────────────────
```
```
$ julia 02.jl
BenchmarkTools.Trial: 
  memory estimate:  1.88 GiB
  allocs estimate:  2340272
  --------------
  minimum time:     1.716 s (5.81% GC)
  median time:      1.759 s (5.76% GC)
  mean time:        1.747 s (5.81% GC)
  maximum time:     1.765 s (5.74% GC)
  --------------
  samples:          3
  evals/sample:     1
```
```
$ julia 02-timed.jl
r[end] = 7.000894875771719
 ───────────────────────────────────────────────────────────────────────────
                                    Time                   Allocations      
                            ──────────────────────   ───────────────────────
      Tot / % measured:          8.58s / 86.7%           2.52GiB / 96.7%    

 Section            ncalls     time   %tot     avg     alloc   %tot      avg
 ───────────────────────────────────────────────────────────────────────────
 Main                    1    7.44s   100%   7.44s   2.43GiB  100%   2.43GiB
   simulation            1    1.79s  24.1%   1.79s   1.88GiB  77.2%  1.88GiB
     loop                1    1.79s  24.1%   1.79s   1.88GiB  77.2%  1.88GiB
       K             60.0k    812ms  10.9%  13.5μs    604MiB  24.3%  10.3KiB
       Pu            60.0k    677ms  9.10%  11.3μs   0.98GiB  40.3%  17.1KiB
       Pp            60.0k    270ms  3.63%  4.50μs    315MiB  12.6%  5.38KiB
   Initialization        1    1.09s  14.6%   1.09s   81.0MiB  3.25%  81.0MiB
 ───────────────────────────────────────────────────────────────────────────
```
```
$ julia 03.jl
BenchmarkTools.Trial: 
  memory estimate:  410.72 MiB
  allocs estimate:  1560290
  --------------
  minimum time:     1.331 s (1.91% GC)
  median time:      1.340 s (1.72% GC)
  mean time:        1.342 s (1.76% GC)
  maximum time:     1.357 s (1.71% GC)
  --------------
  samples:          4
  evals/sample:     1
```
```
$ julia 03-timed.jl
r[end] = 7.000894875771753
 ───────────────────────────────────────────────────────────────────────────
                                    Time                   Allocations      
                            ──────────────────────   ───────────────────────
      Tot / % measured:          8.27s / 86.2%           1.01GiB / 91.7%    

 Section            ncalls     time   %tot     avg     alloc   %tot      avg
 ───────────────────────────────────────────────────────────────────────────
 Main                    1    7.12s   100%   7.12s    952MiB  100%    952MiB
   Simulation            1    1.50s  21.1%   1.50s    390MiB  40.9%   390MiB
     loop                1    1.50s  21.1%   1.50s    389MiB  40.9%   389MiB
       K             60.0k    881ms  12.4%  14.7μs    389MiB  40.9%  6.64KiB
         K-pinv      60.0k    743ms  10.4%  12.4μs    389MiB  40.9%  6.64KiB
       Pu            60.0k    366ms  5.14%  6.10μs     0.00B  0.00%    0.00B
       Pp            60.0k    216ms  3.03%  3.60μs     0.00B  0.00%    0.00B
   Initialization        1    1.10s  15.4%   1.10s   81.0MiB  8.51%  81.0MiB
 ───────────────────────────────────────────────────────────────────────────
```