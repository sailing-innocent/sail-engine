#include "SailScene/node/node3d.h"

namespace sail {

Node3D::Node3D(const Node3D& other) {
	__children = other.__children;
	__parent = other.__parent;
	__is_root = other.__is_root;
	__is_visible = other.__is_visible;
	__transform = other.__transform;
}
Node3D::Node3D(Node3D&& other) {
	__children = std::move(other.__children);
	__parent = std::move(other.__parent);
	__is_root = other.__is_root;
	__is_visible = other.__is_visible;
	__transform = other.__transform;
}

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

void Node3D::set_transform(const math::Transform3D& transform) {
	__transform = transform;
}
const math::Transform3D& Node3D::get_transform() const noexcept {
	return __transform;
}

void Node3D::update() {
	// TODO: update
}

}// namespace sail