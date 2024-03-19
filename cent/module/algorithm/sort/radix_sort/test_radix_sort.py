import pytest 
import numpy as np 

def bitwise_sort(orig_arr, offset, start_N, end_N):
    e = np.zeros_like(orig_arr)
    t = np.zeros_like(orig_arr)
    f = np.zeros_like(orig_arr)
    d = np.zeros_like(orig_arr)
    temp_arr = np.zeros_like(orig_arr)
    next_arr = np.zeros_like(orig_arr)

    lsb = 2 - offset

    for i in range(start_N, end_N):
        # print(f"{orig_arr[i]:032b}")
        bit = (orig_arr[i] >> lsb) & 1
        # temp[i] = not bit
        e[i] = not bit
    print(f"e: {e}") # 0 1 0 1 0
    # prefix sum
    for i in range(start_N + 1, end_N):
        f[i] = f[i-1] + e[i-1]
    total_false = f[end_N-1] + e[end_N-1]

    print(f"f: {f}") # 0 1 1 2 2
    print(f"total_false: {total_false}") # 2

    # move 1 to last and
    for i in range(start_N, end_N):
        t[i] = i - f[i] + total_false
        if (e[i] == 0):
            temp_arr[i] = t[i] # bit == 1, move
        else:
            temp_arr[i] = f[i] + start_N 
    print(f"t: {t}") # 2 3 3 4 4
    print(f"temp_arr: {temp_arr}") # 2 0 3 1 4
    # scatter
    for i in range(start_N, end_N):
        next_arr[temp_arr[i]] = orig_arr[i]

    # 4 2 5 3 1
    for i in range(start_N, end_N):
        orig_arr[i] = next_arr[i]

    return total_false

@pytest.mark.app
def test_radix_sort():
    orig_arr = np.array([5, 4, 3, 2, 1], dtype=np.uint32)

    N = len(orig_arr)
    start_N = 0
    end_N = N 
    split_list = [[start_N, end_N]]
    for offset in range(3):
        print(f"offset: {offset}")     
        split_size = 2 ** offset
        if (offset == 0): 
            split_size = 1
        for i in range(split_size):
            start_N = split_list[split_size - 1 + i][0]
            end_N = split_list[split_size - 1 + i][1]
            print(f"start_N: {start_N}, end_N: {end_N}")
            total_false = bitwise_sort(orig_arr, offset, start_N, end_N)
            print(f"orig_arr: {orig_arr}")
            split_list.append([start_N, total_false + start_N])
            split_list.append([total_false + start_N, end_N])

    
    assert True 