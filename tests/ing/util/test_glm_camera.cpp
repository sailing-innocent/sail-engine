/**
 * @file tests/ing/util/test_glm_camera.cpp
 * @author sailing-innocent
 * @brief The GLM Camera Tester
 * @date 2024-03-13
 */

#include "SailIng/util/camera.h"
#include "SailIng/util/test_helper.h"
#include "test_util.h"
#include <iostream>

namespace sail::test {

int test_glm_camera()
{
  glm::vec3 pos{1.0f, 0.0f, 0.0f};
  glm::vec3 target{0.0f, 0.0f, 0.0f};
  glm::vec3 up{0.0f, 0.0f, 1.0f};
  float fov = 45.0f;
  float aspect = 4.0f / 3.0f;
  float znear = 0.1f;
  float zfar = 100.0f;
  ing::INGFlipZCamera cam(pos, target, up, fov, aspect, znear, zfar);
  auto view_mat = cam.m_view_matrix;
  auto proj_mat = cam.m_proj_matrix;
  glm::vec3 test_pos{0.0f, 1.0f, 1.0f};
  glm::vec3 target_pos{1.0f, 1.0f, -1.0f};
  glm::vec4 result_homo_pos = view_mat * glm::vec4(test_pos, 1.0f);
  glm::vec3 result_pos{result_homo_pos.x / result_homo_pos.w,
                       result_homo_pos.y / result_homo_pos.w,
                       result_homo_pos.z / result_homo_pos.w};

  for (auto i = 0; i < 3; i++) {
    CHECK(target_pos[i] == doctest::Approx(result_pos[i]));
  }
  const float *view_mat_data = (const float *)glm::value_ptr(view_mat);
  for (auto i = 0; i < 16; i++) {
    std::cout << view_mat_data[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}

}  // namespace sail::test

TEST_SUITE("inno::util")
{
  TEST_CASE("glm_camera")
  {
    CHECK(sail::test::test_glm_camera() == 0);
  }
}