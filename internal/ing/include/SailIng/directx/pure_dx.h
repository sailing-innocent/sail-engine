#pragma once
/**
 * @file app/win/dx/dx_pure.h
 * @author sailing-innocent
 * @date 2023-05-03
 * @brief The Basic Pure Win App with DirectX 12
 */

#include "win_app.h"
#include "ing_drx_common.h"

using Microsoft::WRL::ComPtr;

namespace sail::ing {

class SAIL_ING_API INGPureDXApp : public INGWinApp {
public:
	INGPureDXApp(UINT width, UINT height, std::string name);
	virtual ~INGPureDXApp();
	// life cycle
	void init() override;
	bool tick(int count) override;
	void terminate() override;
	void logic_tick() override;
	void render_tick() override;
	// callbacks
	void on_key_down(UINT8) override;
	void on_key_up(UINT8) override {}
	void on_size_changed(UINT width, UINT heightk, bool minimized) override{};

	void get_hardware_adapter(_In_ IDXGIFactory1* p_factory,
							  _Outptr_result_maybenull_ IDXGIAdapter1** pp_adapter,
							  bool request_high_performance_adapter = false);

protected:
	static const UINT m_frame_count = 2;

	ComPtr<ID3D12Device> m_device;
	ComPtr<IDXGISwapChain3> m_swap_chain;
	ComPtr<ID3D12Resource> m_render_targets[m_frame_count];
	ComPtr<ID3D12CommandAllocator> m_command_allocator;
	ComPtr<ID3D12CommandQueue> m_command_queue;
	ComPtr<ID3D12DescriptorHeap> m_rtv_heap;
	ComPtr<ID3D12PipelineState> m_pipeline_state;
	ComPtr<ID3D12GraphicsCommandList> m_command_list;
	UINT m_rtv_descriptor_size;

	// Synchronization objects
	UINT m_frame_index;
	HANDLE m_fence_event;
	ComPtr<ID3D12Fence> m_fence;
	UINT64 m_fence_value;

	// procedure
	virtual void load_pipeline();
	virtual void load_assets();
	virtual void populate_command_list();
	virtual void wait_for_previous_frame();
};

}// namespace sail::ing