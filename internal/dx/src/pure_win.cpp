/**
 * @file app/win/pure.cpp
 * @author sailing-innocent
 * @date 2023-05-02
 * @brief The Implementation of Basic Pure Win App
 */

#include "SailDX/dummy_app/pure_win.h"

namespace sail::dx {

DXPureWinApp::DXPureWinApp(UINT width, UINT height, std::string name) : DXWinApp(width, height, name) {
}

DXPureWinApp::~DXPureWinApp() {
}

void DXPureWinApp::init() {
	// nothing to do
}

bool DXPureWinApp::tick(int count) {
	logic_tick();
	render_tick();
	return true;
}

void DXPureWinApp::logic_tick() {
	// nothing to do
}

void DXPureWinApp::render_tick() {
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(Win32Utils::get_hwnd(), &ps);
	FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));
	EndPaint(Win32Utils::get_hwnd(), &ps);
}

void DXPureWinApp::terminate() {
	// nothing to do
}

void DXPureWinApp::on_key_down(UINT8 key) {
	switch (key) {
		case VK_SPACE:
			MessageBox(0, _T("Hello World!"), _T("Hello"), MB_OK);
			break;
	}
}

}// namespace sail::dx