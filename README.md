# Scalable Evolution Strategies

Implementation of OpenAI's ES in julia using the native threads package and MPI. Based on two papers by uber AI labs [here](https://arxiv.org/abs/1712.06567) and [here](https://arxiv.org/abs/1712.06560).  
This was created because my [python implementation](https://github.com/sash-a/es_pytorch) was too slow, some very informal testing suggest that this repo is 5-6x faster on 8 cores.

### How to run
Example of how to run in `scripts/runner.jl`:

```
julia --project -t 8 scripts/runner.jl
```
