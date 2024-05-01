#include "SailScene/node/node3d.h"

namespace sail {

SceneNode3D::~SceneNode3D() {}
SceneNode3D::SceneNode3D(SceneNode3D&& node) {
	__name = std::move(node.__name);
	__children = std::move(node.__children);
	__parent = std::move(node.__parent);
	__is_root = node.__is_root;
	__is_visible = node.__is_visible;
	__updated = node.__updated;
	m_local_transform = std::move(node.m_local_transform);
	m_global_transform = std::move(node.m_global_transform);
}

SceneNode3D& SceneNode3D::operator=(SceneNode3D&& node) {
	__name = std::move(node.__name);
	__children = std::move(node.__children);
	__parent = std::move(node.__parent);
	__is_root = node.__is_root;
	__is_visible = node.__is_visible;
	__updated = node.__updated;
	m_local_transform = std::move(node.m_local_transform);
	m_global_transform = std::move(node.m_global_transform);
	return *this;
}

void SceneNode3D::set_local_transform(const math::Transform3D& transform) {
	m_local_transform = transform;
	// mark dirty for all child
	for (auto& child : __children) {
		child->mark_dirty();
	}
}
const math::Transform3D& SceneNode3D::get_local_transform() const noexcept {
	return m_local_transform;
}

const math::Transform3D& SceneNode3D::get_global_transform() noexcept {
	if (!__updated) {
		auto parent = get_parent();
		if (parent) {
			m_global_transform = parent->get_global_transform() * m_local_transform;

		} else {
			m_global_transform = m_local_transform;
		}
		__updated = true;
	}
	return m_global_transform;
}

void SceneNode3D::update() {
	// TODO: update
}

}// namespace sail