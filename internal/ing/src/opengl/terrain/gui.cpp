#include "SailIng/opengl/terrain/gui/gui.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

namespace ing::terrain {

GUI::GUI(Window& window) {
	// GUI
	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(window.get_window(), true);
	ImGui_ImplOpenGL3_Init("#version 450");
}

void GUI::draw() {
	SceneElements& scene = *this->scene;

	// forall subscribed object: set gui

	ImGui::Begin("Scene controls: ");
	ImGui::TextColored(ImVec4(1, 1, 0, 1), "Other controls.");
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
				1000.0f / ImGui::GetIO().Framerate,
				ImGui::GetIO().Framerate);
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUI::update() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

GUI::~GUI() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

}// namespace ing::terrain