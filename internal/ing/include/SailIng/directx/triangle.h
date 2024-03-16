#pragma once
/**
 * @file app/win/dx/dx_triangle.h
 * @author sailing-innocent
 * @date 2023-05-04
 * @brief The Basic Pure Win App with DirectX 12
 */

#include "pure_dx.h"

namespace sail::ing {
using Microsoft::WRL::ComPtr;
using namespace DirectX;

class SAIL_ING_API INGTriangleDXApp : public INGPureDXApp {
public:
	INGTriangleDXApp(UINT width, UINT height, std::string name);
	virtual ~INGTriangleDXApp() {}

protected:
	struct Vertex {
		XMFLOAT3 position;
		XMFLOAT4 color;
	};

	// pipeline objects
	CD3DX12_VIEWPORT m_viewport;
	CD3DX12_RECT m_scissor_rect;
	ComPtr<ID3D12RootSignature> m_root_signature;

	// App resources
	ComPtr<ID3D12Resource> m_vertex_buffer;
	D3D12_VERTEX_BUFFER_VIEW m_vertex_buffer_view;

	// procedure
	void load_assets() override;
	void populate_command_list() override;
};

}// namespace sail::ing
