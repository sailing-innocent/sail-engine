/**
 * @file win32_utils.cpp
 * @brief The Win32 Utils
 * @author sailing-innocent
 * @date 2024-05-02
 */
#include "SailDX/dummy_app/win32_utils.h"
namespace sail::dx {

HWND Win32Utils::m_hwnd = nullptr;

int Win32Utils::run(DXWinApp* pApp, HINSTANCE hInstance, int nShowCmd) {
	// parse the command line
	WNDCLASSEX wc;
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = 0;
	wc.lpfnWndProc = WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = _T("BasicWndClass");
	wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	if (!RegisterClassEx(&wc)) {
		MessageBox(NULL, _T("Window Registration Failed!"), _T("Error!"), MB_ICONEXCLAMATION | MB_OK);
	}

	RECT windowRect = {0,
					   0,
					   static_cast<LONG>(pApp->get_width()),
					   static_cast<LONG>(pApp->get_height())};
	AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);
	// Create the window and store a handle to it.
	m_hwnd = CreateWindow(wc.lpszClassName,
						  pApp->get_title(),
						  WS_OVERLAPPEDWINDOW,
						  CW_USEDEFAULT,
						  CW_USEDEFAULT,
						  windowRect.right - windowRect.left,
						  windowRect.bottom - windowRect.top,
						  nullptr,// We have no parent window.
						  nullptr,// We aren't using menus.
						  hInstance,
						  pApp);

	// init the app
	pApp->init();

	ShowWindow(m_hwnd, nShowCmd);
	MSG msg = {};
	while (msg.message != WM_QUIT) {
		// Process any messages in the queue.
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	pApp->terminate();
	return static_cast<char>(msg.wParam);
}

LRESULT CALLBACK Win32Utils::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	if (msg == WM_CREATE) {
		// Save the DXWinApp* passed in to CreateWindow.
		LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
		SetWindowLongPtr(hWnd,
						 GWLP_USERDATA,
						 reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
	}
	DXWinApp* pApp = reinterpret_cast<DXWinApp*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

	switch (msg) {
		case WM_SIZE:
			if (pApp) {
				RECT windiowRect = {};
				GetWindowRect(hWnd, &windiowRect);
				// set window bounds
				pApp->set_window_bounds(windiowRect.left,
										windiowRect.top,
										windiowRect.right,
										windiowRect.bottom);
				RECT clientRect = {};
				GetClientRect(hWnd, &clientRect);
				pApp->on_size_changed(clientRect.right - clientRect.left,
									  clientRect.bottom - clientRect.top,
									  wParam == SIZE_MINIMIZED);
			} else {
				MessageBox(0, _T("pApp NOT EXISTS!"), _T("NOOOO"), MB_OK);
			}
			return 0;
		case WM_LBUTTONDOWN:
			MessageBox(0, _T("Hello World!"), _T("Hello"), MB_OK);
			return 0;
		case WM_KEYDOWN:
			if (pApp) {
				pApp->on_key_down(static_cast<UINT8>(wParam));
			} else {
				MessageBox(0, _T("pApp NOT EXISTS!"), _T("NOOOO"), MB_OK);
			}
			switch (wParam) {
				case VK_ESCAPE:
					DestroyWindow(m_hwnd);
					break;
			}
			return 0;
		case WM_KEYUP:
			if (pApp) {
				pApp->on_key_up(static_cast<UINT8>(wParam));
			} else {
				MessageBox(0, _T("pApp NOT EXISTS!"), _T("NOOOO"), MB_OK);
			}
			return 0;
		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;
		case WM_PAINT: {
			if (pApp) {
				pApp->tick(0);
			} else {
				PAINTSTRUCT ps;
				HDC hdc = BeginPaint(hWnd, &ps);
				FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));
				EndPaint(hWnd, &ps);
			}
			return 0;
		}
	}
	return DefWindowProc(hWnd, msg, wParam, lParam);
}

}// namespace sail::dx