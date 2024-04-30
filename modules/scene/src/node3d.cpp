#include "SailScene/node/node3d.h"

namespace sail {

Node3D::~Node3D() {}

void Node3D::add_child(shared_ptr<Node3D> child) {
	__children.emplace_back(std::move(child));
}

void Node3D::set_parent(shared_ptr<Node3D> parent) {
	__parent = parent;
}

void Node3D::set_visible(bool visible) {
	__is_visible = visible;
}

void Node3D::set_local_transform(const math::Transform3D& transform) {
	m_local_transform = transform;
	// mark dirty for all child
	for (auto& child : __children) {
		child->mark_dirty();
	}
}
const math::Transform3D& Node3D::get_local_transform() const noexcept {
	return m_local_transform;
}

const math::Transform3D& Node3D::get_global_transform() noexcept {
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

void Node3D::update() {
	// TODO: update
}

}// namespace sail