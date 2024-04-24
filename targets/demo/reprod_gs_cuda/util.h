#pragma once
#include <stb_image.h>
#include <stb_image_write.h>
#include <string>
#include <span>

namespace sail::reprod_gs_cuda {
void write_image(std::string_view output_path, int w, int h, std::span<float> odata) noexcept;
}// namespace sail::reprod_gs_cuda