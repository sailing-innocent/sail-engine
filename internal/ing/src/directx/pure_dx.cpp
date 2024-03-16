/**
 * @file directx/pure_dx.h
 * @author sailing-innocent
 * @date 2023-05-03
 * @brief The impl for Basic Pure Win App with DirectX 12
 */

#include "SailIng/directx/pure_dx.h"

namespace sail::ing {

INGPureDXApp::INGPureDXApp(UINT width, UINT height, std::string name)
	: INGWinApp(width, height, name), m_frame_index(0), m_rtv_descriptor_size(0) {
}

INGPureDXApp::~INGPureDXApp() {}

void INGPureDXApp::init() {
	load_pipeline();
	load_assets();
}

void INGPureDXApp::load_pipeline() {
	UINT dxgiFactoryFlags = 0;
	ComPtr<IDXGIFactory4> factory;
	ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

	if (m_use_warp_device) {
		ComPtr<IDXGIAdapter> warp_adapter;
		ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter)));
		ThrowIfFailed(D3D12CreateDevice(
			warp_adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));
	} else {
		ComPtr<IDXGIAdapter1> hardware_adapter;
		get_hardware_adapter(factory.Get(), &hardware_adapter);
		ThrowIfFailed(D3D12CreateDevice(
			hardware_adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));
	}
	// create command queue
	D3D12_COMMAND_QUEUE_DESC queue_desc = {};
	queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowIfFailed(m_device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&m_command_queue)));

	// create swap chain
	DXGI_SWAP_CHAIN_DESC1 swap_chain_desc = {};
	swap_chain_desc.BufferCount = m_frame_count;
	swap_chain_desc.Width = m_width;
	swap_chain_desc.Height = m_height;
	swap_chain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swap_chain_desc.SampleDesc.Count = 1;

	ComPtr<IDXGISwapChain1> swap_chain;
	ThrowIfFailed(factory->CreateSwapChainForHwnd(
		m_command_queue.Get(),
		Win32Utils::get_hwnd(),
		&swap_chain_desc,
		nullptr, nullptr, &swap_chain));

	ThrowIfFailed(factory->MakeWindowAssociation(Win32Utils::get_hwnd(), DXGI_MWA_NO_ALT_ENTER));
	ThrowIfFailed(swap_chain.As(&m_swap_chain));
	m_frame_index = m_swap_chain->GetCurrentBackBufferIndex();

	// create descriptor heaps
	{
		// create rtv descriptor heap
		D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc = {};
		rtv_heap_desc.NumDescriptors = m_frame_count;
		rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
		rtv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		ThrowIfFailed(m_device->CreateDescriptorHeap(&rtv_heap_desc,
													 IID_PPV_ARGS(&m_rtv_heap)));

		m_rtv_descriptor_size =
			m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		/*
        // create dsv descriptor heap
        D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc = {};
        dsv_heap_desc.NumDescriptors = 1;
        dsv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        ThrowIfFailed(m_device->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&m_dsv_heap)));
        */
	}

	// create frame resources
	{
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handle(
			m_rtv_heap->GetCPUDescriptorHandleForHeapStart());
		for (UINT i = 0; i < m_frame_count; i++) {
			ThrowIfFailed(m_swap_chain->GetBuffer(i, IID_PPV_ARGS(&m_render_targets[i])));
			m_device->CreateRenderTargetView(m_render_targets[i].Get(), nullptr, rtv_handle);
			rtv_handle.Offset(1, m_rtv_descriptor_size);
		}
	}

	// create command allocator
	ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
												   IID_PPV_ARGS(&m_command_allocator)));

	// create command list
	ThrowIfFailed(m_device->CreateCommandList(0,
											  D3D12_COMMAND_LIST_TYPE_DIRECT,
											  m_command_allocator.Get(),
											  nullptr,
											  IID_PPV_ARGS(&m_command_list)));
	ThrowIfFailed(m_command_list->Close());

	{
		// create synchronization objects
		ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
		m_fence_value = 1;

		// create an event handle to use for frame synchronization
		m_fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		if (m_fence_event == nullptr) {
			ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
		}
	}
}

void INGPureDXApp::get_hardware_adapter(_In_ IDXGIFactory1* p_factory,
										_Outptr_result_maybenull_ IDXGIAdapter1** pp_adapter,
										bool request_high_performance_adapter) {
	*pp_adapter = nullptr;
	ComPtr<IDXGIAdapter1> adapter;
	ComPtr<IDXGIFactory6> factory6;
	if (SUCCEEDED(p_factory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
		for (UINT adapterIndex = 0; SUCCEEDED(factory6->EnumAdapterByGpuPreference(
				 adapterIndex,
				 request_high_performance_adapter == true ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE : DXGI_GPU_PREFERENCE_UNSPECIFIED,
				 IID_PPV_ARGS(&adapter)));
			 ++adapterIndex) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
				// Don't select the Basic Render Driver adapter.
				// If you want a software adapter, pass in "/warp" on the command line.
				continue;
			}

			// Check to see whether the adapter supports Direct3D 12, but don't create the
			// actual device yet.
			if (SUCCEEDED(D3D12CreateDevice(
					adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
				break;
			}
		}
	}

	if (adapter.Get() == nullptr) {
		for (UINT adapterIndex = 0;
			 SUCCEEDED(p_factory->EnumAdapters1(adapterIndex, &adapter));
			 ++adapterIndex) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
				// Don't select the Basic Render Driver adapter.
				// If you want a software adapter, pass in "/warp" on the command line.
				continue;
			}

			// Check to see whether the adapter supports Direct3D 12, but don't create the
			// actual device yet.
			if (SUCCEEDED(D3D12CreateDevice(
					adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
				break;
			}
		}
	}

	*pp_adapter = adapter.Detach();
}

void INGPureDXApp::load_assets() {
	// plane window, no asset
}

bool INGPureDXApp::tick(int count) {
	logic_tick();
	render_tick();
	return true;
}

void INGPureDXApp::terminate() {
	wait_for_previous_frame();
	CloseHandle(m_fence_event);
}

void INGPureDXApp::logic_tick() {}

void INGPureDXApp::render_tick() {

	populate_command_list();

	ID3D12CommandList* ppCommandLists[] = {m_command_list.Get()};
	m_command_queue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	ThrowIfFailed(m_swap_chain->Present(1, 0));
	wait_for_previous_frame();
}

void INGPureDXApp::populate_command_list() {

	// reset command allocator and command list
	ThrowIfFailed(m_command_allocator->Reset());

	ThrowIfFailed(m_command_list->Reset(m_command_allocator.Get(),
										m_pipeline_state.Get()));

	// buffer barrier: present->render_target
	auto transition = CD3DX12_RESOURCE_BARRIER::Transition(m_render_targets[m_frame_index].Get(),
														   D3D12_RESOURCE_STATE_PRESENT,
														   D3D12_RESOURCE_STATE_RENDER_TARGET);
	m_command_list->ResourceBarrier(
		1,
		&transition);

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handle(
		m_rtv_heap->GetCPUDescriptorHandleForHeapStart(), m_frame_index, m_rtv_descriptor_size);
	// set render target view
	const float clear_color[] = {0.0f, 0.2f, 0.4f, 1.0f};
	m_command_list->ClearRenderTargetView(rtv_handle, clear_color, 0, nullptr);

	// buffer barrier: render_target->present
	transition = CD3DX12_RESOURCE_BARRIER::Transition(m_render_targets[m_frame_index].Get(),
													  D3D12_RESOURCE_STATE_RENDER_TARGET,
													  D3D12_RESOURCE_STATE_PRESENT);
	m_command_list->ResourceBarrier(
		1,
		&transition);

	ThrowIfFailed(m_command_list->Close());
}

void INGPureDXApp::wait_for_previous_frame() {
	// flush
	const UINT64 fence = m_fence_value;
	ThrowIfFailed(m_command_queue->Signal(m_fence.Get(), fence));
	m_fence_value++;

	// wait
	if (m_fence->GetCompletedValue() < fence) {
		ThrowIfFailed(m_fence->SetEventOnCompletion(fence, m_fence_event));
		WaitForSingleObject(m_fence_event, INFINITE);
	}

	m_frame_index = m_swap_chain->GetCurrentBackBufferIndex();
}

// callbacks

void INGPureDXApp::on_key_down(UINT8 key) {
	switch (key) {
		case VK_ESCAPE:
			DestroyWindow(Win32Utils::get_hwnd());
			break;
		case VK_SPACE:
			MessageBox(0, _T("Hello World!"), _T("Hello"), MB_OK);
			break;
	}
}

}// namespace sail::ing
