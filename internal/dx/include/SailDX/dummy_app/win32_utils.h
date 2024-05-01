#pragma once

#include <windows.h>
#include <string>
#include "tchar.h"
#include "win_app.h"

namespace sail::dx {

class DXWinApp;

class SAIL_DX_API Win32Utils {
public:
	static int run(DXWinApp* pApp, HINSTANCE hInstance, int nCmdShow);
	static HWND get_hwnd() { return m_hwnd; }

protected:
	static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

private:
	static HWND m_hwnd;
};

}// namespace sail::dx
