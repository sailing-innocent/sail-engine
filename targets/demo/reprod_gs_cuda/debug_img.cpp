#include <iostream>
#include "util.h"
#include <string>
#include <vector>

#include "SailCu/demo/reprod_gs.h"

int main(int argc, char* argv[]) {
	std::string output_path = "test_stb.png";
	if (argc > 0) {
		output_path = argv[1];
	}
	int w = 256, h = 128;
	std::vector<float> h_data(w * h * 3, 0.0f);
	sail::cu::ReprodGs reprod_gs;
	reprod_gs.debug_img(w, h, h_data);
	sail::reprod_gs_cuda::write_image(output_path, w, h, h_data);
	return 0;
}