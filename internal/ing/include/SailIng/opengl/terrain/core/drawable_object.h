#pragma once
/** 
 * @file terrain/scene/drawable_object.h
 * @author sailing-innocent
 * @date 2023-08-30
 * @brief the terrain drawable object
 */

#include "scene_elements.h"

namespace ing::terrain {

class SAIL_ING_API DrawableObject {
public:
	virtual void draw() = 0;
	virtual void set_gui() {}
	virtual void update() {}

	static SceneElements* scene;
};

}// namespace ing::terrain