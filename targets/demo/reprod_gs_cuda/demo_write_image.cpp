#include <iostream>
#include "util.h"
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
	std::string output_path = "test_stb.png";
	if (argc > 0) {
		output_path = argv[1];
	}
	int w = 256, h = 128;
	std::vector<float> h_data(w * h * 3, 0.0f);
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			h_data[(j * w + i) * 3 + 0] = i / (float)w;
			h_data[(j * w + i) * 3 + 1] = j / (float)h;
			h_data[(j * w + i) * 3 + 2] = 0.0f;
		}
	}
	sail::reprod_gs_cuda::write_image(output_path, w, h, h_data);
	return 0;
}