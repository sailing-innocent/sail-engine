#include "test_util.h"
#include "container/linklist.h"

namespace sail::test {

ListNode* merge_two_sorted_list(ListNode* list1, ListNode* list2) {
	ListNode* p1 = list1;
	ListNode* p2 = list2;

	ListNode* pNewList = new ListNode();
	ListNode* root = pNewList;
	while (p1 != nullptr && p2 != nullptr) {
		if (p1->val < p2->val) {
			pNewList->next = p1;
			p1 = p1->next;
			pNewList = pNewList->next;
		} else {
			pNewList->next = p2;
			p2 = p2->next;
			pNewList = pNewList->next;
		}
	}
	if (p1 != nullptr) {
		pNewList->next = p1;
	}
	if (p2 != nullptr) {
		pNewList->next = p2;
	}
	return root->next;
}

}// namespace sail::test

TEST_CASE("lc_021") {
	using namespace sail::test;
	ListNode n1(1);
	ListNode n2(2);
	ListNode n3(3);
	n1.next = &n2;
	n2.next = &n3;

	ListNode n1_(1);
	ListNode n2_(2);
	ListNode n3_(3);

	n1_.next = &n2_;
	n2_.next = &n3_;

	ListNode* pNew = merge_two_sorted_list(&n1_, &n1);
	REQUIRE(pNew->val == 1);
	REQUIRE(pNew->next->val == 1);
}
