/**
 * @file app/win/pure.cpp
 * @author sailing-innocent
 * @date 2023-05-02
 * @brief The Implementation of Basic Pure Win App
 */

#include "SailIng/directx/pure_win.h"

namespace sail::ing {

INGPureWinApp::INGPureWinApp(UINT width, UINT height, std::string name) : INGWinApp(width, height, name) {
}

INGPureWinApp::~INGPureWinApp() {
}

void INGPureWinApp::init() {
	// nothing to do
}

bool INGPureWinApp::tick(int count) {
	logic_tick();
	render_tick();
	return true;
}

void INGPureWinApp::logic_tick() {
	// nothing to do
}

void INGPureWinApp::render_tick() {
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(Win32Utils::get_hwnd(), &ps);
	FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));
	EndPaint(Win32Utils::get_hwnd(), &ps);
}

void INGPureWinApp::terminate() {
	// nothing to do
}

void INGPureWinApp::on_key_down(UINT8 key) {
	switch (key) {
		case VK_SPACE:
			MessageBox(0, _T("Hello World!"), _T("Hello"), MB_OK);
			break;
	}
}

}// namespace sail::ing