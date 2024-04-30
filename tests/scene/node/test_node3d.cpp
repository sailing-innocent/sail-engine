#include "test_util.h"
#include "SailScene/node/node3d.h"
#include "SailContainer/stl.h"

namespace sail::test {

int test_node_3d() {
	Node3D node;
	node.set_name("RootNode");
	Node3D child1;
	auto p_child1 = make_shared<Node3D>(child1);
	p_child1->set_name("SubNode1");
	Node3D child2;
	auto p_child2 = make_shared<Node3D>(child2);
	p_child2->set_name("SubNode2");

	node.add_child(p_child1);
	node.add_child(p_child2);

	CHECK(node.num_childs() == 2);

	CHECK(node.get_name() == "RootNode");
	CHECK(node[0].get_name() == "SubNode1");
	CHECK(node[1].get_name() == "SubNode2");

	return 0;
}

}// namespace sail::test

TEST_SUITE("scene") {
	TEST_CASE("node3d") {
		CHECK(sail::test::test_node_3d() == 0);
	}
}