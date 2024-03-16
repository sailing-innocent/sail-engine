#include "SailIng/directx/pure_dx.h"

using namespace sail;
_Use_decl_annotations_ int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, PSTR, int nShowCmd) {
	ing::INGPureDXApp app{1280, 720, "Pure DX App"};
	return ing::Win32Utils::run(&app, hInstance, nShowCmd);
}
