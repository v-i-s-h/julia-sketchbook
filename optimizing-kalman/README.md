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

$ julia 01-timed.jl
 ──────────────────────────────────────────────────────────────────
                           Time                   Allocations      
                   ──────────────────────   ───────────────────────
 Tot / % measured:      2.65s / 72.1%           2.35GiB / 99.4%    

 Section   ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────
 K          60.0k    880ms  46.2%  14.7μs    604MiB  25.2%  10.3KiB
 Pu         60.0k    726ms  38.1%  12.1μs   1.29GiB  55.0%  22.5KiB
 Pp         60.0k    300ms  15.7%  5.00μs    472MiB  19.7%  8.06KiB
 ──────────────────────────────────────────────────────────────────r[end] = 7.000894875771719

$ julia 02.jl
BenchmarkTools.Trial: 
  memory estimate:  1.88 GiB
  allocs estimate:  2340272
  --------------
  minimum time:     1.824 s (5.43% GC)
  median time:      1.851 s (5.37% GC)
  mean time:        1.846 s (5.45% GC)
  maximum time:     1.863 s (5.34% GC)
  --------------
  samples:          3
  evals/sample:     1

$ julia 03.jl
BenchmarkTools.Trial: 
  memory estimate:  410.72 MiB
  allocs estimate:  1560290
  --------------
  minimum time:     1.362 s (1.63% GC)
  median time:      1.365 s (1.64% GC)
  mean time:        1.369 s (1.66% GC)
  maximum time:     1.382 s (1.62% GC)
  --------------
  samples:          4
  evals/sample:     1

$ julia 03-timed.jl
 ─────────────────────────────────────────────────────────────────────
                              Time                   Allocations      
                      ──────────────────────   ───────────────────────
   Tot / % measured:       8.45s / 87.2%           1.01GiB / 91.9%    

 Section      ncalls     time   %tot     avg     alloc   %tot      avg
 ─────────────────────────────────────────────────────────────────────
 main              1    7.36s   100%   7.36s    949MiB  100%    949MiB
   K           60.0k    881ms  12.0%  14.7μs    389MiB  41.0%  6.64KiB
     K-pinv    60.0k    737ms  10.0%  12.3μs    389MiB  41.0%  6.64KiB
   Pu          60.0k    370ms  5.03%  6.17μs     0.00B  0.00%    0.00B
   Pp          60.0k    207ms  2.81%  3.45μs     0.00B  0.00%    0.00B
 ─────────────────────────────────────────────────────────────────────r[end] = 7.000894875771753
```