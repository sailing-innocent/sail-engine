// 3D Geometry Transform
#include <EASTL/vector.h>

namespace sail {

class Transform {
public:
	Transform();
	virtual ~Transform();

protected:
	eastl::vector<Transform> m_sub_transform;
};

class Rotation : public Transform {
public:
	Rotation();
	~Rotation();

	// Rotation Display
	enum class RotationType : int {
		Matrix,
		Quaternion,
		AxisAngle
	};
};

class Translation : public Transform {
public:
	Translation();
	~Translation();
};

class ViewTransform : public Transform {
public:
	ViewTransform();
	~ViewTransform();
};

}// namespace sail