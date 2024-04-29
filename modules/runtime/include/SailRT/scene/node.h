#pragma once

/**
 * @file node.h
 * @brief SceneNode Definition
 * @author sailing-innocent
 * @date 2024-04-29
 */

#include "SailRT/runtime.h"
#include <EASTL/vector.h>
#include <EASTL/unique_ptr.h>

namespace sail::runtime {

class Transform;

class SAIL_RT_API SceneNode {
public:
	SceneNode();
	virtual ~SceneNode();
	// delete copy
	SceneNode(const SceneNode&) = delete;
	SceneNode& operator=(const SceneNode&) = delete;

	// delete move
	SceneNode(SceneNode&&) = delete;
	SceneNode& operator=(SceneNode&&) = delete;

private:
	eastl::vector<eastl::unique_ptr<SceneNode>> m_children;
	// TODO hash
	eastl::unique_ptr<Transform> m_transform;
};

}// namespace sail::runtime