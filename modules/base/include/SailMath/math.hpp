#pragma once
/**
 * @file math.hpp
 * @brief The GLM wrapper
 * @author sailing-innocent
 * @date 2024-04-30
 */
#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/ext/matrix_transform.hpp>

namespace sail {
namespace math {
using vec4 = glm::vec4;
using vec3 = glm::vec3;
using mat4 = glm::mat4;
using quat = glm::quat;

template<typename T>
T mat4_cast(const quat& q) {
	return glm::mat4_cast(q);
}

template<typename T>
T translate(const T& m, const vec3& v) {
	return glm::translate(m, v);
}
// scale
template<typename T>
T scale(const T& m, const vec3& v) {
	return glm::scale(m, v);
}

}}// namespace sail::math