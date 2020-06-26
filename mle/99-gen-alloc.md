# Pre-Allocation

If you are intending to use pre-allocated variables, use `mul!()` instead of `*`.
Also, the preallocated variable should be of the same type as the operations output.

# Results

## Case 1
If `y` is pre-allocated, but with not type information
```
y = Vector(undef, size(X, 1))
```
Here `y` becomes pf type `Any`, but `mul!()` return typeof `X` or `β`. Hence,
new allocation is happening in `generate3!()`.
**Output**
```
 ──────────────────────────────────────────────────────────────────────────
                                   Time                   Allocations      
                           ──────────────────────   ───────────────────────
     Tot / % measured:          1.72s / 31.3%            240MiB / 67.8%    

 Section           ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────
 generate3!()         100    492ms  91.4%  4.92ms    156MiB  95.7%  1.56MiB
   mul!(y, X, β)      100    466ms  86.5%  4.66ms    153MiB  93.9%  1.53MiB
   p                  100   20.3ms  3.77%   203μs   3.06MiB  1.88%  31.3KiB
   y                  100   5.91ms  1.10%  59.1μs   7.81KiB  0.00%    80.0B
 generate2!()         100   28.6ms  5.31%   286μs   5.37MiB  3.30%  55.0KiB
   p                  100   19.4ms  3.60%   194μs   3.06MiB  1.88%  31.3KiB
   y                  100   4.60ms  0.85%  46.0μs   7.81KiB  0.00%    80.0B
   y .= X * β         100   4.29ms  0.80%  42.9μs   2.30MiB  1.42%  23.6KiB
 generate1!()         100   17.5ms  3.26%   175μs   1.55MiB  0.96%  15.9KiB
   z = X * β          100   9.19ms  1.71%  91.9μs    794KiB  0.48%  7.94KiB
   p                  100   4.79ms  0.89%  47.9μs    794KiB  0.48%  7.94KiB
   y                  100   3.18ms  0.59%  31.8μs     0.00B  0.00%    0.00B
 ──────────────────────────────────────────────────────────────────────────
```

## Case 2
Pre-allocate `y` with type of `X` or `β`.
```
y = Vector{eltype(X)}(undef, size(X, 1))
```

**Output**
```
 ──────────────────────────────────────────────────────────────────────────
                                   Time                   Allocations      
                           ──────────────────────   ───────────────────────
     Tot / % measured:          1.22s / 2.52%           81.0MiB / 4.79%    

 Section           ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────
 generate2!()         100   10.7ms  34.8%   107μs   1.55MiB  40.0%  15.9KiB
   p                  100   5.13ms  16.7%  51.3μs    794KiB  20.0%  7.94KiB
   y                  100   2.92ms  9.50%  29.2μs     0.00B  0.00%    0.00B
   y .= X * β         100   2.43ms  7.93%  24.3μs    794KiB  20.0%  7.94KiB   <<< Allocation!
 generate1!()         100   10.4ms  33.9%   104μs   1.55MiB  40.0%  15.9KiB
   p                  100   5.16ms  16.8%  51.6μs    794KiB  20.0%  7.94KiB
   y                  100   2.97ms  9.66%  29.7μs     0.00B  0.00%    0.00B
   z = X * β          100   2.09ms  6.82%  20.9μs    794KiB  20.0%  7.94KiB
 generate3!()         100   9.62ms  31.3%  96.2μs    796KiB  20.0%  7.96KiB
   p                  100   5.15ms  16.8%  51.5μs    794KiB  20.0%  7.94KiB
   y                  100   2.93ms  9.55%  29.3μs     0.00B  0.00%    0.00B
   mul!(y, X, β)      100   1.34ms  4.37%  13.4μs     0.00B  0.00%    0.00B   <<< No Allocation!!
 ──────────────────────────────────────────────────────────────────────────
```