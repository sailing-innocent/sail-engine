#pragma once
/** 
 * @file terrain/gui/gui.h
 * @author sailing-innocent
 * @date 2023-08-30
 * @brief the terrain gui object
 */

#include "../core/window.h"
#include "../core/drawable_object.h"
#include <imgui.h>

namespace ing::terrain {
class SAIL_ING_API GUI : public DrawableObject {
public:
	GUI(Window& w);
	~GUI();
	void draw() override;
	void update() override;
	// subscribe
private:
	// subscribers
};
}// namespace ing::terrain
