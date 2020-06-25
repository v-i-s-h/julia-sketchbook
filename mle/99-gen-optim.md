# Results
```
julia> benchmark()
 ─────────────────────────────────────────────────────────────────────────────────
                                          Time                   Allocations      
                                  ──────────────────────   ───────────────────────
         Tot / % measured:             40.6s / 100%            26.7GiB / 100%     

 Section                  ncalls     time   %tot     avg     alloc   %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────
 length()+@inbounds          100    12.7s  31.1%   127ms   8.57GiB  32.1%  87.7MiB
   func4                     100    6.34s  15.6%  63.4ms   4.28GiB  16.1%  43.9MiB
     dot + slice           5.00M    5.73s  14.1%  1.15μs   4.25GiB  15.9%     912B
       slice               5.00M    2.84s  6.99%   568ns   4.17GiB  15.6%     896B
       dot                 5.00M    1.39s  3.43%   279ns   76.3MiB  0.28%    16.0B
   func4!                    100    6.31s  15.5%  63.1ms   4.28GiB  16.1%  43.9MiB
     func4!                  100    6.30s  15.5%  63.0ms   4.25GiB  15.9%  43.5MiB
       dot + slice         5.00M    5.70s  14.0%  1.14μs   4.25GiB  15.9%     912B
         slice             5.00M    2.77s  6.81%   554ns   4.17GiB  15.6%     896B
         dot               5.00M    1.43s  3.52%   286ns   76.3MiB  0.28%    16.0B
     allocate                100   10.4ms  0.03%   104μs   38.2MiB  0.14%   391KiB
 eachindex()                 100    11.6s  28.5%   116ms   8.57GiB  32.1%  87.7MiB
   func2                     100    6.00s  14.8%  60.0ms   4.28GiB  16.1%  43.9MiB
   func2!                    100    5.60s  13.8%  56.0ms   4.28GiB  16.1%  43.9MiB
     func2!                  100    5.56s  13.7%  55.6ms   4.25GiB  15.9%  43.5MiB
     allocate                100   35.9ms  0.09%   359μs   38.2MiB  0.14%   391KiB
 length+@inbounds+@view      100    7.89s  19.4%  78.9ms    687MiB  2.51%  6.87MiB
   func5!                    100    3.97s  9.77%  39.7ms    343MiB  1.26%  3.43MiB
     func5!                  100    3.96s  9.73%  39.6ms    305MiB  1.12%  3.05MiB
       dot + @view         5.00M    3.43s  8.43%   686ns    305MiB  1.12%    64.0B
         dot()             5.00M    1.66s  4.09%   332ns   76.3MiB  0.28%    16.0B
         @view             5.00M    384ms  0.95%  76.8ns    229MiB  0.84%    48.0B
     allocate                100   14.4ms  0.04%   144μs   38.2MiB  0.14%   391KiB
   func5                     100    3.92s  9.66%  39.2ms    343MiB  1.26%  3.43MiB
     dot + @view           5.00M    3.38s  8.32%   676ns    305MiB  1.12%    64.0B
       dot()               5.00M    1.66s  4.08%   331ns   76.3MiB  0.28%    16.0B
       @view               5.00M    370ms  0.91%  74.0ns    229MiB  0.84%    48.0B
 length()                    100    7.21s  17.7%  72.1ms   8.57GiB  32.1%  87.7MiB
   func3                     100    3.78s  9.31%  37.8ms   4.28GiB  16.1%  43.9MiB
   func3!                    100    3.43s  8.43%  34.3ms   4.28GiB  16.1%  43.9MiB
     func3!                  100    3.42s  8.40%  34.2ms   4.25GiB  15.9%  43.5MiB
     allocate                100   10.5ms  0.03%   105μs   38.2MiB  0.14%   391KiB
 Direct Multiplication       100    1.29s  3.17%  12.9ms    305MiB  1.12%  3.05MiB
   func1!                    100    663ms  1.63%  6.63ms    153MiB  0.56%  1.53MiB
     func1!                  100    644ms  1.58%  6.44ms    114MiB  0.42%  1.14MiB
     allocate                100   18.8ms  0.05%   188μs   38.2MiB  0.14%   391KiB
   func1                     100    625ms  1.54%  6.25ms    153MiB  0.56%  1.53MiB
 ─────────────────────────────────────────────────────────────────────────────────
```

# Observations
1. Direct multiplication seems to be the best performing option
2. When `size()` or `length()` is used, performance jumps between a very high delay 
and low delay on alternative loops.