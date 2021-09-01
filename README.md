# Scalable Evolution Strategies

Implementation of OpenAI's ES in julia using MPI. Based on two papers by uber AI labs [here](https://arxiv.org/abs/1712.06567) and [here](https://arxiv.org/abs/1712.06560).  
This was created because my python implementation was too slow, some very informal testing suggest that this repo is 5-6x faster on 8 cores.

### How to run
Example of how to run in `runner.jl`. With MPI installed one can simply run:
```
mpiexec -n 8 julia --project src/runner.jl
```
