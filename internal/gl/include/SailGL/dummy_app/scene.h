#pragma
/**
 * @file scene.h
 * @brief The Dummy Scene App
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/shader/program.h"
#include "SailGL/dummy_app/pure.h"
#include "SailDummy/util/mesh.h"
#include "SailDummy/util/camera.h"

#include <vector>
#include <memory>
#include <string>

namespace sail::gl {

using std::string_view;
using std::unique_ptr;
using std::vector;

class SAIL_GL_API GLSceneDummyApp : public GLPureDummyApp {
public:
	GLSceneDummyApp();
	~GLSceneDummyApp() {
		destroy_buffers();
		destroy_window();
	}
	void load_mesh(string_view mesh_path);
	bool tick(int count = 0) override;
	void init_buffers() override;

protected:
	vector<float> m_vertices;
	vector<unsigned int> m_indices;
	unsigned int VAO, VBO, EBO;
	unique_ptr<ShaderProgram> mp_shader_program;

private:
	vector<dummy::MeshData> m_meshes;
	unique_ptr<dummy::Camera> mp_camera;
};// class INGGLSceneApp

}// namespace sail::gl
