#pragma
/**
 * @file app/opengl/gl_scene_app.h
 * @author sailing-innocent
 * @date 2023-12-15
 * @brief The GLSceneApp
 */

#include "SailIng/opengl/pure_app.h"
#include "SailIng/opengl/shader_program.h"
#include "SailIng/util/mesh_loader.h"
#include "SailIng/util/camera.h"

#include <vector>
#include <memory>
#include <string>

namespace sail::ing {

class SAIL_ING_API INGGLSceneApp : public INGGLPureApp {
public:
	INGGLSceneApp();
	~INGGLSceneApp() {
		destroy_buffers();
		destroy_window();
	}
	void load_mesh(std::string mesh_path);

public:
	bool tick(int count = 0) override;
	void init_buffers() override;

protected:
	std::vector<float> m_vertices;
	std::vector<unsigned int> m_indices;
	unsigned int VAO, VBO, EBO;
	std::unique_ptr<ing::ogl::ShaderProgram> mp_shader_program;

private:
	ing::MeshLoader m_mesh_loader;
	std::vector<MeshData> m_meshes;
	std::shared_ptr<INGFlipZCamera> mp_camera;
};// class INGGLSceneApp

}// namespace sail::ing
