#include "test_util.h"

#include <marl/defer.h>
#include <marl/event.h>
#include <marl/scheduler.h>
#include <marl/waitgroup.h>
#include <cstdio>

namespace sail::test {
int test_marl() {
	// create a marl sheduler
	marl::Scheduler scheduler(marl::Scheduler::Config::allCores());
	scheduler.bind();
	defer(scheduler.unbind());

	constexpr int num_tasks = 10;
	// create an event
	marl::Event say_hello(marl::Event::Mode::Manual);
	marl::WaitGroup said_hello(num_tasks);

	// allocate shedule tasks asynchronously
	for (int i = 0; i < num_tasks; i++) {
		marl::schedule([=] {
			// decrement the wait group when task done
			defer(said_hello.done());
			printf("Task %d waiting to say hello!\n", i);
			say_hello.wait();
			printf("Task %d says hello!\n", i);
		});
	}

	// signal the event
	say_hello.signal();
	// wait for all tasks to say hello
	said_hello.wait();

	// done
	printf("All tasks said hello!\n");

	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::deps") {
	TEST_CASE("marl") {
		REQUIRE(sail::test::test_marl() == 0);
	}
}