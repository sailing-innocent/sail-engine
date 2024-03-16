#pragma once
/**
 * @file helper/device_parallel.h
 * @author sailing-innocent
 * @brief The device parallel
 * @date 2023-12-28
 */

#include "SailInno/core/runtime.h"
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/builtin.h>

namespace sail::inno {

class SAIL_INNO_API DeviceParallel {
	using IntType = int;	// 4 byte
	using FloatType = float;// 4 byte
	template<typename T>
	using Buffer = luisa::compute::Buffer<T>;
	template<typename T>
	using BufferView = luisa::compute::BufferView<T>;
	// shared mem
	template<typename T>
	using SmemType = luisa::compute::Shared<T>;
	template<typename T>
	using SmemTypePtr = luisa::compute::Shared<T>*;
	using Device = luisa::compute::Device;
	using CommandList = luisa::compute::CommandList;
	template<size_t I, typename... Ts>
	using Shader = luisa::compute::Shader<I, Ts...>;

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

public:
	// API
	void scan_exclusive_sum(size_t& temp_storage_size,
							BufferView<IntType> d_in,
							BufferView<IntType> d_out,
							IntType init_v,
							size_t num_item);
	void scan_exclusive_sum(CommandList& cmdlist,
							BufferView<IntType> temp_buffer,
							BufferView<IntType> d_in,
							BufferView<IntType> d_out,
							IntType init_v,
							size_t num_item);

	void scan_inclusive_sum(size_t& temp_storage_size,
							BufferView<IntType> d_in,
							BufferView<IntType> d_out,
							size_t num_item);

	void scan_inclusive_sum(luisa::compute::CommandList& cmdlist,
							BufferView<IntType> temp_buffer,
							BufferView<IntType> d_in,
							BufferView<IntType> d_out, size_t num_item);

private:
	void compile(Device& device);
	void compile_reduce_shaders(Device& device);
	void compile_scan_shaders(Device& device);
	void compile_radix_sort_shaders(Device& device);
	void get_temp_size(size_t& temp_storage_size, size_t num_item);
	luisa::compute::Int conflict_free_offset(luisa::compute::Int i) { return i >> m_log_mem_banks; }

	void prescan_array_recursive_int(
		CommandList& cmdlist,
		BufferView<IntType> temp_storage,
		BufferView<IntType> arr_in, BufferView<IntType> arr_out,
		size_t num_elements, int offset, int level) noexcept;

private:
	// shader
	// prescan int
	U<Shader<1, int, int, Buffer<IntType>, Buffer<IntType>, Buffer<IntType>, int, int, int>> ms_prescan_int = nullptr;
	// uniform add int
	U<Shader<1, Buffer<IntType>, Buffer<IntType>, int, int, int>> ms_uniform_add_int = nullptr;

	U<Shader<1, Buffer<IntType>, IntType>> ms_add_int = nullptr;
};

}// namespace sail::inno