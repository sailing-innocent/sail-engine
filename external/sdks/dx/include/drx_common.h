#pragma once
/**
* @file drx_common.h
* @brief The common header for DirectX
*/

// d3d12
#include <DirectXMath.h>// for XMVector, XMFloat, XMFloat4
#include <comdef.h>		// for _com_error
#include <d3d12.h>		// for D3D12 interface
#include <dxgi1_6.h>	// for DXGI interface
#include <D3Dcompiler.h>// for D3DCompileFromFile
#include <wrl.h>		// for Microsoft::WRL::ComPTr
#include "d3dx12.h"
#include <iostream>
#include <stdexcept>

inline std::string Hr2String(HRESULT hr) {
	char s_str[64] = {};
	sprintf_s(s_str, "HRESULT of 0x%08X", static_cast<UINT>(hr));
	return std::string(s_str);
}

class HrException : public std::runtime_error {
public:
	HrException(HRESULT hr)
		: std::runtime_error(Hr2String(hr)), m_hr(hr) {
	}
	HRESULT Error() const { return m_hr; }

private:
	const HRESULT m_hr;
};

inline void ThrowIfFailed(HRESULT hr) {
	if (FAILED(hr)) {
		// throw HrException(hr);
	}
}

static constexpr DXGI_FORMAT g_BackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
static constexpr DXGI_FORMAT g_DepthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
static constexpr UINT g_iSwapChainBufferCount = 2;
