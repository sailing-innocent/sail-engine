// leetcode 024. Swap the Pairs in linklist
#include "test_util.h"
#include <vector>

#include "container/linklist.h"

namespace sail::test {

void build_list(ListNode* head, std::vector<int> arr) {
	ListNode* cur = head;
	if (arr.size() == 0) {
		return;
	}
	cur->val = arr[0];
	for (auto i = 1; i < arr.size(); i++) {
		cur->next = new ListNode(arr[i]);
		cur = cur->next;
	}
}

void check_list(ListNode* head, std::vector<int> arr) {
	ListNode* cur = head;
	for (auto item : arr) {
		REQUIRE(cur->val == item);
		cur = cur->next;
	}
}

ListNode* swap_pairs(ListNode* head) {
	ListNode* cur = head;
	ListNode* prev = cur;
	ListNode* pprev = nullptr;
	while (cur != nullptr) {
		cur = cur->next;
		if (cur != nullptr) {
			// swap
			if (pprev != nullptr) {
				pprev->next = cur;
			} else {
				head = cur;
			}
			prev->next = cur->next;
			cur->next = prev;
			cur = prev->next;
			pprev = prev;
			prev = cur;
		}
	}
	return head;
}

void clear_list_node(ListNode* head) {
	ListNode* cur = head;
	while (cur != nullptr) {
		ListNode* tmp = cur;
		cur = cur->next;
		delete tmp;
	}
}

int test_swap(std::vector<int> in_arr, std::vector<int> target_arr) {
	ListNode* head = new ListNode();
	build_list(head, in_arr);
	head = swap_pairs(head);
	check_list(head, target_arr);
	clear_list_node(head);
	return 0;
}

}// namespace sail::test

TEST_CASE("lc_024") {
	using namespace sail::test;
	test_swap({1, 2, 3, 4}, {2, 1, 4, 3});
	test_swap({}, {});
	test_swap({1}, {1});
	test_swap({1, 2}, {2, 1});
}
