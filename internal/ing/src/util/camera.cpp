
/**
 * @file util/camera.cpp
 * @brief The ING Camera Utils
 * @date 2023-09-29
 * @author sailing-innocent
 */
#include "SailIng/util/camera.h"
#include <iostream>

namespace sail::ing {

INGFlipZCamera::INGFlipZCamera(const glm::vec3 &pos,
                               const glm::vec3 &target,
                               const glm::vec3 &up,
                               const float fov,
                               const float aspect,
                               const float znear,
                               const float zfar)
{
  m_dir = glm::normalize(target - pos);
  m_right = glm::normalize(glm::cross(m_dir, up));
  m_up = glm::cross(m_right, m_dir);

  m_cam_pos = pos;
  m_tan_half_fov = glm::tan(glm::radians(fov / 2.0f));
  m_aspect = aspect;
  m_znear = znear;
  m_zfar = zfar;

  m_view_matrix = glm::lookAt(pos, target, up);
  m_proj_matrix = glm::perspective(fov, aspect, znear, zfar);
}

INGFlipYCamera::INGFlipYCamera(const glm::vec3 &pos,
                               const glm::vec3 &target,
                               const glm::vec3 &up,
                               const float fov,  // means fovy
                               const float aspect,
                               const float znear,
                               const float zfar)
{
  m_z = glm::normalize(target - pos);
  m_x = glm::normalize(glm::cross(m_z, up));
  m_y = glm::cross(m_z, m_x);

  std::cout << "m_x: " << m_x.x << " " << m_x.y << " " << m_x.z << std::endl;
  std::cout << "m_y: " << m_y.x << " " << m_y.y << " " << m_y.z << std::endl;
  std::cout << "m_z: " << m_z.x << " " << m_z.y << " " << m_z.z << std::endl;

  m_cam_pos = pos;
  std::cout << "m_cam_pos: " << m_cam_pos.x << " " << m_cam_pos.y << " " << m_cam_pos.z
            << std::endl;
  m_tan_half_fov = glm::tan(glm::radians(fov / 2.0f));
  m_aspect = aspect;
  m_znear = znear;
  m_zfar = zfar;

  m_view_matrix = glm::mat4(0.0f);
  m_view_matrix[0][0] = m_x.x;
  m_view_matrix[1][0] = m_x.y;
  m_view_matrix[2][0] = m_x.z;
  m_view_matrix[0][1] = m_y.x;
  m_view_matrix[1][1] = m_y.y;
  m_view_matrix[2][1] = m_y.z;
  m_view_matrix[0][2] = m_z.x;
  m_view_matrix[1][2] = m_z.y;
  m_view_matrix[2][2] = m_z.z;
  m_view_matrix[3][0] = -glm::dot(m_x, m_cam_pos);
  m_view_matrix[3][1] = -glm::dot(m_y, m_cam_pos);
  m_view_matrix[3][2] = -glm::dot(m_z, m_cam_pos);
  m_view_matrix[3][3] = 1.0f;

  // test
  std::cout << "translate: " << -glm::dot(m_x, m_cam_pos) << " " << -glm::dot(m_y, m_cam_pos)
            << " " << -glm::dot(m_z, m_cam_pos) << " " << std::endl;
  glm::vec4 orig = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
  glm::vec4 result = m_view_matrix * orig;
  std::cout << "result: " << result.x << " " << result.y << " " << result.z << " " << result.w
            << std::endl;

  // m_proj_matrix = glm::perspective(fov, aspect, znear, zfar);
  m_proj_matrix = glm::mat4(1.0f);
  auto top = znear * m_tan_half_fov;
  auto bottom = -top;
  auto right = top * aspect;
  auto left = -right;
  auto z_sign = 1.0;
  // simply orthognal
  m_proj_matrix[0][0] = 2.0 * znear / (right - left);
  m_proj_matrix[1][1] = 2.0 * znear / (top - bottom);
  // perspective
  m_proj_matrix[2][0] = (right + left) / (right - left);
  m_proj_matrix[2][1] = (top + bottom) / (top - bottom);
  m_proj_matrix[2][3] = z_sign;
  m_proj_matrix[2][2] = z_sign * zfar / (zfar - znear);
  m_proj_matrix[3][2] = -(z_sign * znear) / (zfar - znear);

  m_proj_matrix = m_proj_matrix * m_view_matrix;
}

};  // namespace sail::ing