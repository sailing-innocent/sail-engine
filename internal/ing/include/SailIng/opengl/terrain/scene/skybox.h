#pragma once
/** 
 * @file terrain/scene/skybox.h
 * @author sailing-innocent
 * @date 2023-10-22
 * @brief the terrain skybox
 */

#include "../core/drawable_object.h"
#include "../core/buffers.h"

namespace ing::terrain {

class SAIL_ING_API SkyBox : public DrawableObject {
public:
	// Volumetric Clouds
	SkyBox();
	~SkyBox();
	virtual void draw();
	virtual void set_gui();
	virtual void update();

	unsigned int get_texture() {
		return m_fbo->tex;
	}

private:
	FrameBufferObject* m_fbo;
};

}// namespace ing::terrain