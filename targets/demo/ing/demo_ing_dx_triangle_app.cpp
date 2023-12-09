#include <SailIng/directx/triangle.h>

using namespace sail;
_Use_decl_annotations_ int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, PSTR, int nShowCmd) {
	ing::INGTriangleDXApp app{1280, 720, "Triangle DX App"};
	return ing::Win32Utils::run(&app, hInstance, nShowCmd);
}
