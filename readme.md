# Sail Engine

Sail Engine 是将我之前积攒下来代码的合并，利用动态库将之前独立的项目代码组合成一个小引擎。

- doc: markdown + latex 构建的一个编译系统
- cent: py脚本上层
- internal/inno: 基于LuisaCompute的一些功能
- internal/ing: 基于dx/vk/cuda/opengl的一些图形API封装尝试

HighLight
- internal/inno/diff_render/中实现了一个GaussianSplatting的LC复现
- internal/inno/solver/csigsph/ 实现了一个SPH的流体模拟 可以通过 demo_inno_fluid_sph 示例查看