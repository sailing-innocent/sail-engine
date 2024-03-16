#include "SailIng/directx/triangle.h"

namespace sail::ing {

INGTriangleDXApp::INGTriangleDXApp(UINT width, UINT height, std::string name)
	: INGPureDXApp(width, height, name)// set frame index & rtv_descriptor_size
	  ,
	  m_viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)), m_scissor_rect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)) {
	// get asset path
	// aspect ratio
}

void INGTriangleDXApp::load_assets() {
	// create empty root signature
	{
		CD3DX12_ROOT_SIGNATURE_DESC root_signature_desc;
		root_signature_desc.Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
		ComPtr<ID3DBlob> signature;
		ComPtr<ID3DBlob> error;
		ThrowIfFailed(D3D12SerializeRootSignature(
			&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
		ThrowIfFailed(m_device->CreateRootSignature(0,
													signature->GetBufferPointer(),
													signature->GetBufferSize(),
													IID_PPV_ARGS(&m_root_signature)));
	}

	// prepare shaders
	{
		ComPtr<ID3DBlob> vertex_shader;
		ComPtr<ID3DBlob> pixel_shader;

		UINT compile_flags = 0;

		ThrowIfFailed(D3DCompileFromFile(L"assets/shaders/hlsl/basic.hlsl",
										 nullptr,
										 nullptr,
										 "VSMain",
										 "vs_5_0",
										 compile_flags,
										 0,
										 &vertex_shader,
										 nullptr));
		ThrowIfFailed(D3DCompileFromFile(L"assets/shaders/hlsl/basic.hlsl",
										 nullptr,
										 nullptr,
										 "PSMain",
										 "ps_5_0",
										 compile_flags,
										 0,
										 &pixel_shader,
										 nullptr));

		D3D12_INPUT_ELEMENT_DESC input_element_desc[] = {
			{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
			{"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

		D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = {};
		pso_desc.InputLayout = {input_element_desc, _countof(input_element_desc)};
		pso_desc.pRootSignature = m_root_signature.Get();
		pso_desc.VS = CD3DX12_SHADER_BYTECODE(vertex_shader.Get());
		pso_desc.PS = CD3DX12_SHADER_BYTECODE(pixel_shader.Get());
		pso_desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		pso_desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		pso_desc.DepthStencilState.DepthEnable = FALSE;
		pso_desc.DepthStencilState.StencilEnable = FALSE;
		pso_desc.SampleMask = UINT_MAX;
		pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		pso_desc.NumRenderTargets = 1;
		pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		pso_desc.SampleDesc.Count = 1;

		ThrowIfFailed(m_device->CreateGraphicsPipelineState(
			&pso_desc, IID_PPV_ARGS(&m_pipeline_state)));
	}
	// create vertex buffer
	{
		// position, color
		Vertex triangle_vertices[] = {
			{{0.0f, 0.25f * m_aspect_ratio, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
			{{0.25f, -0.25f * m_aspect_ratio, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
			{{-0.25f, -0.25f * m_aspect_ratio, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}}};

		const UINT vertex_buffer_size = sizeof(triangle_vertices);

		auto heap_property = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
		auto buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(vertex_buffer_size);
		ThrowIfFailed(m_device->CreateCommittedResource(
			&heap_property,
			D3D12_HEAP_FLAG_NONE, &buffer_desc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_vertex_buffer)));

		// copy the triangle data to vertex buffer
		UINT8* p_vertex_data_begin;
		CD3DX12_RANGE read_range(0, 0);
		ThrowIfFailed(m_vertex_buffer->Map(
			0, &read_range, reinterpret_cast<void**>(&p_vertex_data_begin)));
		memcpy(p_vertex_data_begin, triangle_vertices, sizeof(triangle_vertices));
		m_vertex_buffer->Unmap(0, nullptr);

		// initialize vertex buffer view
		m_vertex_buffer_view.BufferLocation = m_vertex_buffer->GetGPUVirtualAddress();
		m_vertex_buffer_view.StrideInBytes = sizeof(Vertex);
		m_vertex_buffer_view.SizeInBytes = vertex_buffer_size;
	}
}

void INGTriangleDXApp::populate_command_list() {
	// reset command allocator and command list
	ThrowIfFailed(m_command_allocator->Reset());
	ThrowIfFailed(m_command_list->Reset(m_command_allocator.Get(),
										m_pipeline_state.Get()));

	// set root signature
	m_command_list->SetGraphicsRootSignature(m_root_signature.Get());
	m_command_list->RSSetViewports(1, &m_viewport);
	m_command_list->RSSetScissorRects(1, &m_scissor_rect);

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
	m_command_list->OMSetRenderTargets(1, &rtv_handle, FALSE, nullptr);

	const float clear_color[] = {0.3f, 0.2f, 0.4f, 1.0f};
	m_command_list->ClearRenderTargetView(rtv_handle, clear_color, 0, nullptr);
	// draw primitives
	m_command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	m_command_list->IASetVertexBuffers(0, 1, &m_vertex_buffer_view);
	m_command_list->DrawInstanced(3, 1, 0, 0);

	// buffer barrier: render_target->present
	transition = CD3DX12_RESOURCE_BARRIER::Transition(m_render_targets[m_frame_index].Get(),
													  D3D12_RESOURCE_STATE_RENDER_TARGET,
													  D3D12_RESOURCE_STATE_PRESENT);
	m_command_list->ResourceBarrier(
		1, &transition);

	ThrowIfFailed(m_command_list->Close());
}

}// namespace sail::ing
