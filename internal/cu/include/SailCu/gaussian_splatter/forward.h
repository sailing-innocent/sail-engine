#pragma once

namespace sail::cu::gs::FORWARD {

void preprocess(int P);
void render(const dim3 grid, const dim3 block, float* out_color);

}// namespace sail::cu::gs::FORWARD