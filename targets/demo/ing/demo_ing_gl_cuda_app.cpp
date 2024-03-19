#include "SailIng/cuda/basic_app.h"
#include "cuda.h"

int main() {
	sail::ing::INGCUDAApp app{
		"cuda app",
		800,
		600};
	app.init();
	while (app.tick()) {
	}
	return 0;
}