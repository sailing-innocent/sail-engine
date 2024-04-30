#pragma once
/**
 * @file scene.h
 * @brief The Scene Tree for Sail Engine
 * @author sailing-innocent
 * @date 2024-04-30
 */
#include "SailScene/config.h"
#include "SailContainer/stl.h"// for vector, unique_ptr, shared_ptr

namespace sail {

class Node3D;
class SAIL_SCENE_API Scene {
	shared_ptr<Node3D> __root;

public:
	Scene();
	~Scene();
	void update();
	void render();
};

}// namespace sail