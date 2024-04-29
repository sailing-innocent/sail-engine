#pragma once
#include "SailRT/runtime.h"

#include <EASTL/unique_ptr.h>
#include <EASTL/shared_ptr.h>
// the sene

namespace sail::runtime {

class SceneNode;
class SAIL_RT_API Scene {
public:
	Scene();
	virtual ~Scene();

private:
	eastl::unique_ptr<SceneNode> m_root = nullptr;
};

}// namespace sail::runtime