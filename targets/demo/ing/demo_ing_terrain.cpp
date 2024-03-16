#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "SailIng/opengl/terrain/core/scene_elements.h"
#include "SailIng/opengl/terrain/core/window.h"
#include "SailIng/opengl/terrain/core/buffers.h"
#include "SailIng/opengl/terrain/core/drawable_object.h"
#include "SailIng/opengl/terrain/gui/gui.h"
#include "SailIng/opengl/terrain/scene/camera.h"
#include "SailIng/opengl/terrain/scene/terrain.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

using namespace ing::terrain;

int main() {
	glm::vec3 start_position(0.0f, 800.f, 0.0f);
	Camera camera(start_position);
	int success;
	Window window(success, 1600, 900);
	if (!success) {
		return -1;
	}
	// bind camera to window for input handling
	// window.camera = &camera;

	GUI gui{window};

	glm::mat4 proj = glm::perspective(
		glm::radians(camera.zoom),
		(float)Window::SCR_WIDTH / (float)Window::SCR_HEIGHT,
		5.f, 10000000.0f);
	SceneElements scene;
	scene.cam = &camera;
	scene.proj_matrix = proj;
	std::cout << "proj matrix: " << std::endl;
	for (int i = 0; i < 4; ++i) {
		std::cout << proj[i][0] << " " << proj[i][1] << " " << proj[i][2] << " "
				  << proj[i][3] << std::endl;
	}

	// show view matrix
	glm::mat4 view = camera.get_view_matrix();
	std::cout << "view matrix: " << std::endl;
	for (int i = 0; i < 4; ++i) {
		std::cout << view[i][0] << " " << view[i][1] << " " << view[i][2] << " "
				  << view[i][3] << std::endl;
	}

	DrawableObject::scene = &scene;

	glm::vec3 fogColor(0.5, 0.6, 0.7);

	int grid_length = 120;
	Terrain terrain(grid_length);

	// gui.subscribe terrain

	while (window.continueLoop()) {
		// handle input
		float frame_time = 0.0f;
		window.process_input(frame_time);

		// update state
		// terrain.update_tiles_position

		// gui.update();

		// draw
		glEnable(GL_DEPTH_TEST);

		glClearColor(fogColor[0], fogColor[1], fogColor[2], 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		terrain.up = 1.0f;
		terrain.draw();
		// gui.draw();

		// swap buffer and poll events
		window.swap_buffers_and_poll_events();
	}
	return 0;
}
