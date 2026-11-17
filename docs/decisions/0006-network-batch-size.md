# 0006. The network runs on two windows at a time on the processor, apart from bacpipe's batch size
Status: Accepted
Date: 2026-11-17

## Context

bacpipe groups the windows of a recording into batches of `global_batch_size` (default 16) and runs the network on each batch. The first browser runs passed bacpipe's batches straight to ONNX Runtime Web. On the processor (WebAssembly), BirdNET+ V3.0 was then extremely slow: about 48 minutes for 61 windows. With one window per call it embedded about 3 windows a second.

## Options considered

1. **Keep bacpipe's batch for the network.**
   - Loses: BirdNET+ V3.0 is not usable in the page at the default setting.
2. **Change bacpipe's default in the page** (`global_batch_size` 1 or 2).
   - Loses: the setting would no longer be bacpipe's default, and a visitor who sets it back gets the slow path again.
3. **Split each of bacpipe's batches into small calls of the network.** bacpipe still makes its batches. The worker runs the network on two windows at a time and gives bacpipe the outputs of the whole batch.
   - Gains: speed, and `global_batch_size` keeps its meaning for bacpipe.
   - Needs: proof that the numbers do not depend on the split.

## Decision

Option 3. `CPU_BATCH = 2` in `app/worker.js`. On the graphics card (WebGPU, experimental) the worker uses batches of 4, padded to that size, because WebGPU compiles its programs for one batch shape.

## Evidence

- The replay session gives bacpipe its outputs window by window. bacpipe's code is the same whatever the split.
- In the browser test, with calls of 2 windows against bacpipe's batches of 16, the embeddings agreed with bacpipe's to within 0.000005.
- Three recordings of one minute at 32 kHz, both models, on a laptop processor with 4 threads: 37 s for the whole run. Perch 2.0 embedded 4.6 windows a second and BirdNET+ V3.0 3.6 a second. On bacpipe's test data (7 recordings) the two models embedded 4.4 and 4.2 windows a second. Later repeats on the same laptop, one of them on battery, were several times slower, so speed varies with the state of the machine.

## Consequences

- **Easier:** both models are usable at bacpipe's default settings.
- **Harder:** the reason for the number 2 is a measurement on one machine. Another browser or a later ONNX Runtime Web may prefer another size.
- **Reversal:** one constant.
