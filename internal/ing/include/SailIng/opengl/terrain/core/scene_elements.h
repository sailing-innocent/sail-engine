#pragma once
/** 
 * @file terrain/scene/buffers.h
 * @author sailing-innocent
 * @date 2023-08-30
 * @brief the terrain scene element
 */

#include "../scene/camera.h"
#include "buffers.h"

namespace ing::terrain
{

struct SceneElements
{
    glm::vec3          light_pos, light_color, light_dir, fog_color, seed;
    glm::mat4          proj_matrix;
    Camera*            cam;
    FrameBufferObject* scene_FBO;
};

}  // namespace ing::terrain