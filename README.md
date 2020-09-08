# Intro

An implementation of a feedforward ANN with optimization via backpropagation and genetic algorithm.

Developed using [Eclipse CDT](https://www.eclipse.org/cdt/) with [Nsight Eclipse Plugins](https://docs.nvidia.com/cuda/nsight-eclipse-plugins-guide/index.html).

# Init

Clone the repo and submodules:

```
git clone --recurse-submodules https://github.com/MajesticThrust-KPFU-2019/cuda-neural-network-project.git
```

Or:

```
git clone https://github.com/MajesticThrust-KPFU-2019/cuda-neural-network-project.git
cd cuda-neural-network-project
git submodule update --init --recursive
```

If using the Eclipse IDE with Nsight plugin, the project folder must be in the root of the workspace, so as to not break the includes from submodules.