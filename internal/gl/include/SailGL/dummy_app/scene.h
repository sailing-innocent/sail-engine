#pragma
/**
 * @file scene.h
 * @brief The Dummy Scene App
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/dummy_app/pure.h"
#include "SailGL/shader/program.h"
#include "SailGL/util/mesh_loader.h"
#include "SailGL/util/camera.h"

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
