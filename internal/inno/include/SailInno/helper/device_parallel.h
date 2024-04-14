#pragma once
/**
 * @file helper/device_parallel.h
 * @author sailing-innocent
 * @brief The device parallel
 * @date 2023-12-28
 */
#include <SailInno/core/runtime.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/rhi/resource.h>
#include <type_traits>
#include "SailInno/util/math/calc.h"// for imax

namespace sail::inno {

template<typename T>
static constexpr bool is_numeric_v = std::is_integral_v<T> || std::is_floating_point_v<T>;
template<typename T>
concept NumericT = is_numeric_v<T>;

class SAIL_INNO_API DeviceParallel : public LuisaModule {
	using IntType = int;	// 4 byte
	using FloatType = float;// 4 byte

	template<NumericT Type4Byte>
	using ReduceShaderT = Shader<1,
								 Buffer<Type4Byte>,
								 Buffer<Type4Byte>,
								 int, int, int, int>;

public:
	int m_block_size = 256;
	int m_num_banks = 32;
	// shared_mem_banks = 2 ^ log_mem_banks
	int m_log_mem_banks = 5;

private:
	size_t m_shared_mem_size = 0;

public:
	// lifecycle
	void create(Device& device);

	// API
	template<NumericT Type4Byte>
	void scan_exclusive_sum(
		size_t& temp_storage_size,
		BufferView<Type4Byte> d_in,
		BufferView<Type4Byte> d_out,
		Type4Byte init_v,
		size_t num_item) {
		get_temp_size(temp_storage_size, num_item);
	}

	template<NumericT Type4Byte>
	void scan_exclusive_sum(
		CommandList& cmdlist,
		BufferView<Type4Byte> temp_buffer,
		BufferView<Type4Byte> d_in,
		BufferView<Type4Byte> d_out,
		Type4Byte init_v,
		size_t num_item) {
		size_t temp_storage_size = 0;
		get_temp_size(temp_storage_size, num_item);
		LUISA_ASSERT(temp_storage_size <= temp_buffer.size(), "temp_buffer size is not enough");
		prescan_array_recursive<Type4Byte>(
			cmdlist,
			temp_buffer, d_in, d_out,
			num_item, 0, 0);
		// add for all // brute force
		std::string_view key = luisa::compute::Type::of<Type4Byte>()->description();
		auto* ms_add_it = ms_add_map.find(key);
		auto ms_add_ptr = reinterpret_cast<Shader<1, Buffer<Type4Byte>, Type4Byte>*>(&(*ms_add_it->second));
		cmdlist << (*ms_add_ptr)(d_out, init_v).dispatch(num_item);
	}

	template<NumericT Type4Byte>
	void reduce(size_t& temp_storage_size,
				BufferView<Type4Byte> d_in,
				BufferView<Type4Byte> d_out,
				size_t num_item, int op = 0) {
		get_temp_size(temp_storage_size, num_item);
	}
	template<NumericT Type4Byte>
	void reduce(
		CommandList& cmdlist,
		BufferView<Type4Byte> temp_buffer,
		BufferView<Type4Byte> d_in,
		BufferView<Type4Byte> d_out,
		size_t num_item, int op = 0) {
		size_t temp_storage_size = 0;
		get_temp_size(temp_storage_size, num_item);
		LUISA_ASSERT(temp_buffer.size() >= temp_storage_size, "Please resize the Temp Buffer.");
		reduce_array_recursive<Type4Byte>(cmdlist, temp_buffer, d_in, d_out, num_item, 0, 0, op);
	}

private:
	template<NumericT Type4Byte>
	void compile(Device& device) {
		using namespace luisa;
		using namespace luisa::compute;

		const auto n_blocks = m_block_size;
		size_t shared_mem_size = m_shared_mem_size;
		// see scanRootToLeavesInt in prefix_sum.cu
		auto scan_root_to_leaves = [&](SmemTypePtr<Type4Byte>& s_data, Int stride, Int n) {
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

		auto clear_last_element = [&](Int storeSum, SmemTypePtr<Type4Byte>& s_data, BufferVar<Type4Byte>& g_blockSums, Int blockIndex) {
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
		auto build_sum = [&](SmemTypePtr<Type4Byte>& s_data, Int n) -> Int {
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

		auto prescan_block = [&](Int storeSum, SmemTypePtr<Type4Byte>& s_data, BufferVar<Type4Byte>& blockSums, Int blockIndex, Int n) {
			$if(blockIndex == 0) { blockIndex = Int(block_id().x); };
			Int stride = build_sum(s_data, n);// build the sum in place up the tree
			clear_last_element(storeSum, s_data, blockSums, blockIndex);
			scan_root_to_leaves(s_data, stride, n);// traverse down tree to build the scan
		};

		auto load_shared_chunk_from_mem =
			[&](Int isNP2,
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

		auto load_shared_chunk_from_mem_op = [&](SmemTypePtr<Type4Byte>& s_data, BufferVar<Type4Byte>& g_idata,
												 Int n, Int& baseIndex, Int op) {
			Int threadIdx_x = Int(thread_id().x);
			Int blockIdx_x = Int(block_id().x);
			Int blockDim_x = Int(block_size().x);

			Int thid = threadIdx_x;
			Int mem_ai = baseIndex + threadIdx_x;
			Int mem_bi = mem_ai + blockDim_x;

			Int ai = thid;
			Int bi = thid + blockDim_x;
			Int bankOffsetA = conflict_free_offset(ai);// compute spacing to avoid bank conflicts
			Int bankOffsetB = conflict_free_offset(bi);

			Var<Type4Byte> initial;
			$if(op == 0) { initial = Type4Byte(0); }							// sum
			$elif(op == 1) { initial = std::numeric_limits<Type4Byte>::min(); } // max
			$elif(op == 2) { initial = std::numeric_limits<Type4Byte>::max(); };// min

			Var<Type4Byte> data_ai = initial;
			Var<Type4Byte> data_bi = initial;

			$if(ai < n) { data_ai = g_idata.read(mem_ai); };// Cache the computational window in shared memory pad values beyond n with zeros
			$if(bi < n) { data_bi = g_idata.read(mem_bi); };
			(*s_data)[ai + bankOffsetA] = data_ai;
			(*s_data)[bi + bankOffsetB] = data_bi;
		};

		auto build_op = [&](SmemTypePtr<Type4Byte>& s_data, Int n, Int op) {
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
					$if(op == 0) { (*s_data)[bi] += (*s_data)[ai]; }
					$elif(op == 1) { (*s_data)[bi] = max((*s_data)[bi], (*s_data)[ai]); }
					$elif(op == 2) { (*s_data)[bi] = min((*s_data)[bi], (*s_data)[ai]); };
				};
				stride *= 2;
				d = d >> 1;
			};
		};

		auto reduce_block = [&](
								SmemTypePtr<Type4Byte>& s_data,
								BufferVar<Type4Byte>& block_sums,
								Int block_index,
								Int n, Int op) {
			$if(block_index == 0) { block_index = Int(block_id().x); };
			build_op(s_data, n, op);// build the op in place up the tree
			clear_last_element(1, s_data, block_sums, block_index);
		};

		// see storeSharedChunkToMem in prefix_sum.cu
		auto store_shared_chunk_to_mem =
			[&](Int isNP2, BufferVar<Type4Byte>& g_odata,
				SmemTypePtr<Type4Byte>& s_data,
				Int n, Int ai, Int bi, Int mem_ai, Int mem_bi, Int bankOffsetA, Int bankOffsetB) {
			sync_block();
			// write results to global memory
			$if(ai < n) { g_odata.write(mem_ai, (*s_data)[ai + bankOffsetA]); };
			$if(bi < n) { g_odata.write(mem_bi, (*s_data)[bi + bankOffsetB]); };
		};

		luisa::string_view key = Type::of<Type4Byte>()->description();
		luisa::unique_ptr<Shader<1,
								 int, int,
								 Buffer<Type4Byte>,
								 Buffer<Type4Byte>,
								 Buffer<Type4Byte>,
								 int, int, int>>
			ms_prescan = nullptr;
		lazy_compile(device, ms_prescan,
					 [&](Int storeSum, Int isNP2,// bool actually
						 BufferVar<Type4Byte> g_idata,
						 BufferVar<Type4Byte> g_odata,
						 BufferVar<Type4Byte> g_blockSums,
						 Int n, Int blockIndex, Int baseIndex) {
			set_block_size(n_blocks);
			Int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			Int blockIdx_x = Int(block_id().x);
			Int blockDim_x = Int(block_size().x);
			SmemTypePtr<Type4Byte> s_dataInt = new SmemType<Type4Byte>{shared_mem_size};
			$if(baseIndex == 0) { baseIndex = blockIdx_x * (blockDim_x << 1); };
			load_shared_chunk_from_mem(isNP2, s_dataInt, g_idata, n, baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
			prescan_block(storeSum, s_dataInt, g_blockSums, blockIndex, n);
			store_shared_chunk_to_mem(isNP2, g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
		});
		ms_prescan_map.try_emplace(key, std::move(ms_prescan));

		luisa::unique_ptr<Shader<1,
								 Buffer<Type4Byte>,
								 Buffer<Type4Byte>,
								 int, int, int>>
			ms_uniform_add = nullptr;

		// see uniformAddInt in prefix_sum.cu
		lazy_compile(device, ms_uniform_add,
					 [&](BufferVar<Type4Byte> g_data,
						 BufferVar<Type4Byte> uniforms,
						 Int n, Int blockOffset, Int baseIndex) {
			set_block_size(n_blocks);

			luisa::compute::Shared<Type4Byte> uni{1};
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

		ms_uniform_add_map.try_emplace(key, std::move(ms_uniform_add));

		luisa::unique_ptr<
			Shader<1, Buffer<Type4Byte>, Type4Byte>>
			ms_add = nullptr;
		lazy_compile(device, ms_add,
					 [](luisa::compute::BufferVar<Type4Byte> buf, luisa::compute::Var<Type4Byte> v) {
			auto idx = dispatch_id().x;
			auto val = buf.read(idx) + v;
			buf.write(idx, val);
		});
		ms_add_map.try_emplace(
			luisa::compute::Type::of<Type4Byte>()->description(), std::move(ms_add));

		luisa::unique_ptr<ReduceShaderT<Type4Byte>>
			ms_reduce = nullptr;
		lazy_compile(device, ms_reduce, [&](BufferVar<Type4Byte> g_idata, BufferVar<Type4Byte> g_block_sums, Int n, Int block_index, Int base_index, Int op) {
			set_block_size(n_blocks);
			Int ai, bi, mem_ai, mem_bi, bank_offset_a, bank_offset_b;
			Int block_id_x = Int(block_id().x);
			Int block_dim_x = Int(block_size().x);

			SmemTypePtr<Type4Byte> s_data = new SmemType<Type4Byte>{shared_mem_size};
			$if(base_index == 0) {
				base_index = block_id_x * (block_dim_x << 1);
			};
			load_shared_chunk_from_mem_op(s_data, g_idata, n, base_index, op);
			reduce_block(s_data, g_block_sums, block_index, n, op);
		});

		ms_reduce_map.try_emplace(
			luisa::compute::Type::of<Type4Byte>()->description(), std::move(ms_reduce));
	}

	void compile_reduce_shaders(Device& device);
	void compile_radix_sort_shaders(Device& device);
	void get_temp_size(size_t& temp_storage_size, size_t num_item);
	luisa::compute::Int conflict_free_offset(luisa::compute::Int i) { return i >> m_log_mem_banks; }

	template<NumericT Type4Byte>
	void prescan_array_recursive(
		CommandList& cmdlist,
		BufferView<Type4Byte> temp_storage,
		BufferView<Type4Byte> arr_in,
		BufferView<Type4Byte> arr_out,
		size_t num_elements, int offset, int level) noexcept {
		using namespace inno::math;
		int block_size = m_block_size;
		int num_blocks = imax(1, (int)ceil((float)num_elements / (2.0f * block_size)));
		int num_threads;

		if (num_blocks > 1) {
			num_threads = block_size;
		} else if (is_power_of_two(num_elements)) {
			num_threads = num_elements / 2;
		} else {
			num_threads = floor_pow_2(num_elements);
		}

		int num_elements_per_block = num_threads * 2;
		int num_elements_last_block =
			num_elements - (num_blocks - 1) * num_elements_per_block;
		int num_threads_last_block = imax(1, num_elements_last_block / 2);
		int np2_last_block = 0;
		int shared_mem_last_block = 0;

		if (num_elements_last_block != num_elements_per_block) {
			// NOT POWER OF 2
			np2_last_block = 1;
			if (!is_power_of_two(num_elements_last_block)) {
				num_threads_last_block = floor_pow_2(num_elements_last_block);
			}
		}

		size_t size_elements = temp_storage.size() - offset;
		BufferView<Type4Byte> temp_buffer_level =
			temp_storage.subview(offset, size_elements);

		// execute the scan
		auto key = luisa::compute::Type::of<Type4Byte>()->description();
		auto* ms_prescan_it = ms_prescan_map.find(key);
		auto ms_prescan_ptr = reinterpret_cast<
			Shader<1,
				   int, int,
				   Buffer<Type4Byte>,
				   Buffer<Type4Byte>,
				   Buffer<Type4Byte>,
				   int, int, int>*>(&(*ms_prescan_it->second));

		auto* ms_uniform_add_it = ms_uniform_add_map.find(key);
		auto ms_uniform_add_ptr = reinterpret_cast<
			Shader<1,
				   Buffer<Type4Byte>,
				   Buffer<Type4Byte>,
				   int, int, int>*>(&(*ms_uniform_add_it->second));

		if (num_blocks > 1) {
			// recursive
			cmdlist << (*ms_prescan_ptr)(true, false,
										 arr_in,
										 arr_out,
										 temp_buffer_level,
										 num_elements_per_block, 0, 0)
						   .dispatch(block_size * (num_blocks - np2_last_block));

			if (np2_last_block) {
				// Last Block
				cmdlist << (*ms_prescan_ptr)(
							   true, true,
							   arr_in, arr_out, temp_buffer_level,
							   num_elements_last_block, num_blocks - 1,
							   num_elements - num_elements_last_block)
							   .dispatch(block_size);
			}

			prescan_array_recursive<Type4Byte>(
				cmdlist,
				temp_buffer_level, temp_buffer_level,
				temp_buffer_level,
				num_blocks, num_blocks, level + 1);

			cmdlist << (*ms_uniform_add_ptr)(
						   arr_out, temp_buffer_level,
						   num_elements - num_elements_last_block,
						   0, 0)
						   .dispatch(block_size * (num_blocks - np2_last_block));

			if (np2_last_block) {
				cmdlist << (*ms_uniform_add_ptr)(
							   arr_out, temp_buffer_level,
							   num_elements_last_block, num_blocks - 1,
							   num_elements - num_elements_last_block)
							   .dispatch(block_size);
			}
		} else if (is_power_of_two(num_elements)) {
			// non-recursive
			cmdlist << (*ms_prescan_ptr)(
						   false, false,
						   arr_in, arr_out, temp_buffer_level,
						   num_elements, 0, 0)
						   .dispatch(block_size);
		} else {
			// non-recursive
			cmdlist << (*ms_prescan_ptr)(
						   false, true,
						   arr_in, arr_out, temp_buffer_level,
						   num_elements, 0, 0)
						   .dispatch(block_size);
		}
	}

	template<NumericT Type4Byte>
	void reduce_array_recursive(
		luisa::compute::CommandList& cmdlist,
		BufferView<Type4Byte> temp_storage,
		BufferView<Type4Byte> arr_in,
		BufferView<Type4Byte> arr_out,
		int num_elements,
		int offset, int level, int op) noexcept {

		using namespace inno::math;
		int block_size = m_block_size;
		int num_blocks = imax(1, (int)ceil((float)num_elements / (2.0f * block_size)));
		int num_threads;

		if (num_blocks > 1) {
			num_threads = block_size;
		} else if (is_power_of_two(num_elements)) {
			num_threads = num_elements / 2;
		} else {
			num_threads = floor_pow_2(num_elements);
		}

		int num_elements_per_block = num_threads * 2;
		int num_elements_last_block =
			num_elements - (num_blocks - 1) * num_elements_per_block;
		int num_threads_last_block = imax(1, num_elements_last_block / 2);
		int np2_last_block = 0;
		int shared_mem_last_block = 0;

		if (num_elements_last_block != num_elements_per_block) {
			// NOT POWER OF 2
			np2_last_block = 1;
			if (!is_power_of_two(num_elements_last_block)) {
				num_threads_last_block = floor_pow_2(num_elements_last_block);
			}
		}

		size_t size_elements = temp_storage.size() - offset;
		BufferView<Type4Byte> temp_buffer_level =
			temp_storage.subview(offset, size_elements);

		// execute the scan
		auto key = luisa::compute::Type::of<Type4Byte>()->description();
		auto* ms_reduce_it = ms_reduce_map.find(key);
		auto ms_reduce_ptr = reinterpret_cast<
			ReduceShaderT<Type4Byte>*>(&(*ms_reduce_it->second));

		if (num_blocks > 1) {
			// recursive
			cmdlist << (*ms_reduce_ptr)(
						   arr_in, temp_buffer_level,
						   num_elements, 0, 0, op)
						   .dispatch(block_size * (num_blocks - np2_last_block));

			if (np2_last_block) {
				// Last Block
				cmdlist << (*ms_reduce_ptr)(
							   arr_in, temp_buffer_level,
							   num_elements_last_block, num_blocks - 1,
							   num_elements - num_elements_last_block, op)
							   .dispatch(block_size);
			}

			reduce_array_recursive<Type4Byte>(
				cmdlist,
				temp_buffer_level, temp_buffer_level,
				arr_out,
				num_blocks, num_blocks, level + 1, op);
		} else {
			// non-recursive
			cmdlist << (*ms_reduce_ptr)(
						   arr_in, temp_buffer_level,
						   num_elements, 0, 0, op)
						   .dispatch(block_size);
			cmdlist << arr_out.copy_from(temp_buffer_level);
		}
	}

	// for scan
	luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_prescan_map;
	luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_uniform_add_map;
	luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_add_map;
	// for reduce
	luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_map;
};

}// namespace sail::inno