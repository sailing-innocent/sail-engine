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
	vector<shared_ptr<Node3D>> __children;
	shared_ptr<Node3D> __parent = nullptr;
	bool __is_root = false;
	bool __is_visible = true;
	math::Transform3D __transform;

public:
	Node3D() = default;
	virtual ~Node3D();
	Node3D(const Node3D&);
	Node3D(Node3D&&);
	void add_child(shared_ptr<Node3D> child);
	// TODO: remove child
	void set_parent(shared_ptr<Node3D> parent);
	shared_ptr<Node3D> get_parent() const;
	void set_visible(bool visible);
	[[nodiscard]] bool is_visible() const;
	void set_transform(const math::Transform3D& transform);// relative to parent
	[[nodiscard]] const math::Transform3D& get_transform() const noexcept;
	void update();
};

}// namespace sail