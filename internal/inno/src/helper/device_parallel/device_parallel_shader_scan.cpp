/**
 * @file helper/device_parallel_shader_scan.cpp
 * @author sailing-innocent
 * @brief The device parallel scan shader
 * @date 2023-12-28
 */
#include "SailInno/helper/device_parallel.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno {

void DeviceParallel::compile_scan_shaders(Device& device) {
	const auto n_blocks = m_block_size;
	size_t shared_mem_size = m_shared_mem_size;

	// see scanRootToLeavesInt in prefix_sum.cu
	auto scan_root_to_leaves = [&]<typename Type4Byte>(
								   SmemTypePtr<Type4Byte>& s_data, Int stride, Int n) {
		Int thid = Int(thread_id().x);
		Int blockDim_x = Int(block_size().x);

		// traverse down the tree building the scan in place
		Int d = def(1);
		$while(d <= blockDim_x) {
			stride >>= 1;
			sync_block();

			$if(thid < d) {
				Int i = (stride * 2) * thid;
				Int ai = i + stride - 1;
				Int bi = ai + stride;
				ai += conflict_free_offset(ai);
				bi += conflict_free_offset(bi);
				Var<Type4Byte> t = (*s_data)[ai];
				(*s_data)[ai] = (*s_data)[bi];// left child <- root
				(*s_data)[bi] += t;			  // right child <- root + left child
			};
			d = d << 1;
		};
	};

	auto clear_last_element = [&]<typename Type4Byte>(Int storeSum, SmemTypePtr<Type4Byte>& s_data, BufferVar<Type4Byte>& g_blockSums, Int blockIndex) {
		Int thid = Int(thread_id().x);
		Int d = Int(block_size().x);
		$if(thid == 0) {
			Int index = (d << 1) - 1;
			index += conflict_free_offset(index);
			$if(storeSum == 1) {
				// write this block's total sum to the corresponding index in the blockSums array
				g_blockSums.write(blockIndex, (*s_data)[index]);
			};
			(*s_data)[index] = Type4Byte(0);// zero the last element in the scan so it will propagate back to the front
		};
	};
	// see buildSum in prefix_sum.cu
	auto build_sum = [&]<typename Type4Byte>(
						 SmemTypePtr<Type4Byte>& s_data, Int n) -> Int {
		Int thid = Int(thread_id().x);
		Int stride = def(1);

		// build the sum in place up the tree
		Int d = Int(block_size().x);
		$while(d > 0) {
			sync_block();
			$if(thid < d) {
				Int i = (stride * 2) * thid;
				Int ai = i + stride - 1;
				Int bi = ai + stride;
				ai += conflict_free_offset(ai);
				bi += conflict_free_offset(bi);
				(*s_data)[bi] += (*s_data)[ai];
			};
			stride *= 2;
			d = d >> 1;
		};
		return stride;
	};

	auto prescan_block = [&]<typename Type4Byte>(Int storeSum, SmemTypePtr<Type4Byte>& s_data, BufferVar<Type4Byte>& blockSums, Int blockIndex, Int n) {
		$if(blockIndex == 0) { blockIndex = Int(block_id().x); };
		Int stride = build_sum(s_data, n);// build the sum in place up the tree
		clear_last_element(storeSum, s_data, blockSums, blockIndex);
		scan_root_to_leaves(s_data, stride, n);// traverse down tree to build the scan
	};

	auto load_shared_chunk_from_mem =
		[&]<typename Type4Byte>(
			Int isNP2,
			SmemTypePtr<Type4Byte>& s_data, BufferVar<Type4Byte>& g_idata,
			Int n, Int& baseIndex,
			Int& ai, Int& bi, Int& mem_ai, Int& mem_bi,
			Int& bankOffsetA, Int& bankOffsetB) {
		Int threadIdx_x = Int(thread_id().x);
		Int blockIdx_x = Int(block_id().x);
		Int blockDim_x = Int(block_size().x);

		Int thid = threadIdx_x;
		mem_ai = baseIndex + threadIdx_x;
		mem_bi = mem_ai + blockDim_x;

		ai = thid;
		bi = thid + blockDim_x;
		bankOffsetA = conflict_free_offset(ai);// compute spacing to avoid bank conflicts
		bankOffsetB = conflict_free_offset(bi);

		Var<Type4Byte> data_ai = Type4Byte(0);
		Var<Type4Byte> data_bi = Type4Byte(0);

		$if(ai < n) { data_ai = g_idata.read(mem_ai); };// Cache the computational window in shared memory pad values beyond n with zeros
		$if(bi < n) { data_bi = g_idata.read(mem_bi); };
		(*s_data)[ai + bankOffsetA] = data_ai;
		(*s_data)[bi + bankOffsetB] = data_bi;
	};

	// see storeSharedChunkToMem in prefix_sum.cu
	auto store_shared_chunk_to_mem =
		[&]<typename Type4Byte>(
			Int isNP2, BufferVar<Type4Byte>& g_odata,
			SmemTypePtr<Type4Byte>& s_data,
			Int n, Int ai, Int bi, Int mem_ai, Int mem_bi, Int bankOffsetA, Int bankOffsetB) {
		sync_block();
		// write results to global memory
		$if(ai < n) { g_odata.write(mem_ai, (*s_data)[ai + bankOffsetA]); };
		$if(bi < n) { g_odata.write(mem_bi, (*s_data)[bi + bankOffsetB]); };
	};

	lazy_compile(device, ms_prescan_int,
				 [&](Int storeSum, Int isNP2,
					 BufferVar<IntType> g_idata,
					 BufferVar<IntType> g_odata,
					 BufferVar<IntType> g_blockSums,
					 Int n, Int blockIndex, Int baseIndex) {
		set_block_size(n_blocks);
		Int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
		Int blockIdx_x = Int(block_id().x);
		Int blockDim_x = Int(block_size().x);

		SmemTypePtr<IntType> s_dataInt = new SmemType<IntType>{shared_mem_size};

		$if(baseIndex == 0) { baseIndex = blockIdx_x * (blockDim_x << 1); };
		load_shared_chunk_from_mem(isNP2, s_dataInt, g_idata, n, baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
		prescan_block(storeSum, s_dataInt, g_blockSums, blockIndex, n);
		store_shared_chunk_to_mem(isNP2, g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
	});

	// see uniformAddInt in prefix_sum.cu
	lazy_compile(device, ms_uniform_add_int,
				 [&](BufferVar<IntType> g_data,
					 BufferVar<IntType> uniforms,
					 Int n, Int blockOffset, Int baseIndex) {
		set_block_size(n_blocks);

		luisa::compute::Shared<IntType> uni{1};
		Int threadIdx_x = Int(thread_id().x);
		Int blockIdx_x = Int(block_id().x);
		Int blockDim_x = Int(block_size().x);
		$if(threadIdx_x == 0) { uni[0] = uniforms.read(blockIdx_x + blockOffset); };
		Int address = (blockIdx_x * (blockDim_x << 1)) + baseIndex + threadIdx_x;

		sync_block();

		// note two adds per thread
		$if(threadIdx_x < n) {
			g_data.atomic(address).fetch_add(uni[0]);
			$if(threadIdx_x + blockDim_x < n) {
				g_data.atomic(address + blockDim_x).fetch_add(uni[0]);
			};
		};
	});
}

}// namespace sail::inno