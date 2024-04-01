#pragma once

/**
 * @file SailInno/util/graphic/image.h
 * @author sailing-innocent
 * @date 2023-12-30
 * @brief Image Space Transformation
 */

namespace sail::inno::util {

template<typename Float_T>
inline Float_T ndc2pix(Float_T ndc, Float_T resolution) {
	return ((ndc + 1.0f) * resolution - 1.0f) / 2.0f;
}

template<typename Float_T>
inline Float_T pix2ndc(Float_T pix, Float_T resolution) {
	return 2.0f * pix / resolution - 1.0f;
}

}// namespace sail::inno::util
