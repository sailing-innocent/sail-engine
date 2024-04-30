#pragma once
/**
 * @file node3d.h
 * @brief The 3D Scene Node
 * @author sailing-innocent
 * @date 2024-04-30
 */

#include "SailScene/config.h"
#include "SailContainer/stl.h"// for vector, unique_ptr, shared_ptr
#include "SailMath/transform.hpp"

namespace sail {

class SAIL_SCENE_API Node3D {
protected:
	string __name;
	vector<shared_ptr<Node3D>> __children;
	shared_ptr<Node3D> __parent = nullptr;
	bool __is_root = false;
	bool __is_visible = true;
	bool __updated = false;
	math::Transform3D m_local_transform;
	math::Transform3D m_global_transform;

public:
	Node3D() = default;
	virtual ~Node3D();
	Node3D(const Node3D&) = delete;
	Node3D& operator=(const Node3D&) = delete;
	Node3D(Node3D&&) = delete;
	Node3D& operator=(Node3D&&) = delete;

	[[nodiscard]] Node3D& operator[](int index) noexcept {
		return *__children[index];
	}
	void add_child(shared_ptr<Node3D> child);
	// TODO: remove child
	void set_parent(shared_ptr<Node3D> parent);
	shared_ptr<Node3D> get_parent() const;
	void set_visible(bool visible);
	[[nodiscard]] bool is_visible() const;
	void set_local_transform(const math::Transform3D& transform);// relative to parent
	[[nodiscard]] const math::Transform3D& get_local_transform() const noexcept;
	[[nodiscard]] const math::Transform3D& get_global_transform() noexcept;
	[[nodiscard]] int num_childs() const noexcept {
		return static_cast<int>(__children.size());
	}
	[[nodiscard]] bool is_root() const noexcept {
		return __is_root;
	}
	void set_name(const string& name) {
		__name = name;
	}
	[[nodiscard]] const string& get_name() const noexcept {
		return __name;
	}
	void mark_dirty() {
		__updated = false;
	}

	void update();
};

}// namespace sail