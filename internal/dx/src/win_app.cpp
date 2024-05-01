#include "SailDX/dummy_app/win_app.h"
#include "tchar.h"
#include <string>
#include <windows.h>

namespace sail::dx {

DXWinApp::DXWinApp(UINT width, UINT height, std::string name)
	: m_width(width), m_height(height), m_title(name), m_use_warp_device(false), m_aspect_ratio(width / height) {
	// get asset path
	// aspect ratio
}

DXWinApp::~DXWinApp() {}

void DXWinApp::set_window_bounds(int left, int top, int right, int bottom) {
	m_window_bounds.left = left;
	m_window_bounds.top = top;
	m_window_bounds.right = right;
	m_window_bounds.bottom = bottom;
}

}// namespace sail::dx