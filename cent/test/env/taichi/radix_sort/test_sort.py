import pytest 

import taichi as ti 
import numpy as np 

@ti.kernel 
def prefix_sum_internal(t:ti.u32, h:ti.u32, y:ti.template()):
    for i in range(h):
        src = ((i>>t)<<(t+1)) + (1<<t) - 1
        dst = src + 1 + (i & ((1<<t) - 1))
        y[dst] += y[src]

def prefix_sum(x:ti.template(),y:ti.template()):
    n = x.shape[0]
    y.copy_from(x)
    total_step = int(np.log2(n))
    h = n // 2
    for t in range(total_step):
        prefix_sum_internal(t,h,y)


def radix_sort(x: ti.template(), index_map: ti.template()):
    N = x.shape[0]

    zero = ti.field(ti.i8, shape=N)
    zero_sum = ti.field(ti.i32, shape=N)
    tb1 = ti.field(ti.u32, shape=N)
    tb2 = ti.field(ti.u32, shape=N)
    tb1.copy_from(x)
    
    # map_table_init
    for bit_i in range(32):
        # fetch (src,dst) pair
        # zero_count
        # prefix_sum
        # get_map
        pass 



@pytest.mark.current 
def test_radix_sort():
    ti.init(arch=ti.gpu,kernel_profiler=True)

    # generate 1024 random uint [0,64]
    test_data = np.random.randint(0,1<<6,1<<10,dtype=np.uint32)
    assert test_data.shape == (1<<10,) 

    x = ti.field(ti.u32, shape=test_data.shape)
    x.from_numpy(test_data)
    y1 = ti.field(ti.u32, shape=test_data.shape)
    y2 = ti.field(ti.u32, shape=test_data.shape)
    ti.profiler.clear_kernel_profiler_info()
    # prefix sum
    prefix_sum(x,y1)
    ti.profiler.print_kernel_profiler_info()
    ti.profiler.clear_kernel_profiler_info()
    # prefix_sum kernel

    # gpu_data
    # index_map 

    # for i in range(16):
    #    sorted_arr[index_map[i]] = gpu_data[i]


