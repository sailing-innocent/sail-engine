#include "SailMath/math.hpp"
#include "test_util.h"
#include "SailScene/node/node3d.h"
#include "SailContainer/stl.h"
#include "SailMath/transform.hpp"

#include <iostream>

namespace sail::test {

int test_node_3d() {
	math::vec3 v(1.0f, 0.0f, 0.0f);
	math::quat q(1.0f, 0.0f, 0.0f, 0.0f);
	math::vec3 s(1.0f, 1.0f, 1.0f);
	math::Transform3D t(q, v, s);

	auto mat_root = t.matrix();
	for (auto i = 0; i < 4; i++) {
		for (auto j = 0; j < 4; j++) {
			std::cout << mat_root[i][j] << " ";
		}
	}
	std::cout << std::endl;

	Node3D node;
	auto p_node = make_shared<Node3D>(std::forward<Node3D>(node));
	p_node->set_name("RootNode");
	p_node->set_local_transform(t);

	Node3D child1;
	auto p_child1 = make_shared<Node3D>(std::forward<Node3D>(child1));
	p_child1->set_name("SubNode1");
	math::quat q1 = math::angleAxis(90.0f / 180.0f * math::pi<float>(), v);
	math::Transform3D t1(q, v, s);
	p_child1->set_local_transform(t1);

	Node3D child2;
	auto p_child2 = make_shared<Node3D>(std::forward<Node3D>(child2));
	p_child2->set_name("SubNode2");

	p_node->add_child(p_child1);
	p_child1->set_parent(p_node);

	p_node->add_child(p_child2);
	p_child2->set_parent(p_node);

	CHECK(p_node->num_childs() == 2);

	CHECK(p_node->get_name() == "RootNode");
	CHECK((*p_node)[0].get_name() == "SubNode1");
	CHECK((*p_node)[1].get_name() == "SubNode2");

	auto gt1 = p_child1->get_global_transform();
	auto mat = gt1.matrix();
	for (auto i = 0; i < 4; i++) {
		for (auto j = 0; j < 4; j++) {
			std::cout << mat[i][j] << " ";
		}
	}

	return 0;
}

}// namespace sail::test

TEST_SUITE("scene") {
	TEST_CASE("node3d") {
		CHECK(sail::test::test_node_3d() == 0);
	}
}