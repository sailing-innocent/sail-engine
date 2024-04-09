#ifndef CUDA_SH_PROCESSOR_H_INCLUDED
#define CUDA_SH_PROCESSOR_H_INCLUDED

namespace CudaSHProcessor {
class SHProcessor {
public:
	static void forward(
		// input
		const float* shs,
		const float* means3D,
		const float* cam_pos,
		// params
		const int P, int D, int M,
		// output
		float* color);

	static void backward(
		// input
		const float* dL_dcolor,
		// params
		const int P, int D, int M, int R,
		const float* shs,
		const float* means3D,
		const float* cam_pos,
		const float* color
		// output
		float* dL_dshs);
}
}// namespace CudaSHProcessor