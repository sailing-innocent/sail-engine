#include "util.h"
namespace sail::reprod_gs_cuda {
void write_image(std::string_view output_path, int w, int h, std::span<float> odata) noexcept {
	auto* odata_uchar = (unsigned char*)malloc(w * h * 3);
	int off = 0;
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			off = (j * w + i) * 3;
			float u = odata[off + 0];
			float v = odata[off + 1];
			float w = odata[off + 2];
			odata_uchar[off + 0] = (unsigned char)(u * 255);
			odata_uchar[off + 1] = (unsigned char)(v * 255);
			odata_uchar[off + 2] = (unsigned char)(w * 255);
		}
	}
	// h rows and w cols, flip y axis
	stbi_write_png(output_path.data(), w, h, 3, odata_uchar, 0);
	stbi_image_free(odata_uchar);
}
}// namespace sail::reprod_gs_cuda