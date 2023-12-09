#include "SailIng/directx/pure_win.h"

using namespace sail;
_Use_decl_annotations_ int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, PSTR, int nShowCmd) {
	ing::INGPureWinApp app{1280, 720, "Pure Win App"};
	return ing::Win32Utils::run(&app, hInstance, nShowCmd);
}