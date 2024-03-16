#include "SailIng/opengl/terrain/utils/plane.h"
#include <iostream>

namespace ing::terrain {

void initialize_plane_VAO(
	const int res,
	const int width,
	GLuint* plane_VAO,
	GLuint* plane_VBO,
	GLuint* plane_EBO) {
	const int n_points = res * res;
	// generate vertices
	const int size = n_points * 3 + n_points * 3 + n_points * 2;// pos, normal, texcoord
	float* vertices = new float[size];
	for (int i = 0; i < res; i++) {
		for (int j = 0; j < res; j++) {
			// add position
			float x = j * (float)width / (res - 1) - width / 2.0f;
			float y = 0.0f;
			float z = -i * (float)width / (res - 1) + width / 2.0f;

			vertices[(i + j * res) * 8 + 0] = x;
			vertices[(i + j * res) * 8 + 1] = y;
			vertices[(i + j * res) * 8 + 2] = z;

			// add normal (up)
			vertices[(i + j * res) * 8 + 3] = 0.0f;
			vertices[(i + j * res) * 8 + 4] = 1.0f;
			vertices[(i + j * res) * 8 + 5] = 0.0f;

			// add texcoord
			vertices[(i + j * res) * 8 + 6] = (float)j / (res - 1);
			vertices[(i + j * res) * 8 + 7] = (float)(res - i - 1) / (res - 1);
		}
	}

	// show vertices
	// for (int i = 0; i < size; i++) {
	//     std::cout << vertices[i] << " ";
	//     if (i % 8 == 7) {
	//         std::cout << std::endl;
	//     }
	// }

	// generate indices
	const int n_tris = (res - 1) * (res - 1) * 2;
	int* tris_indices = new int[n_tris * 3];

	for (int i = 0; i < n_tris; i++) {
		int tris_per_row = 2 * (res - 1);
		for (int j = 0; j < tris_per_row; j++) {
			if (!(i % 2)) {//upper triangle
				int k = i * 3;
				int triIndex = i % tris_per_row;

				int row = i / tris_per_row;
				int col = triIndex / 2;
				tris_indices[k] = row * res + col;
				tris_indices[k + 1] = ++row * res + col;
				tris_indices[k + 2] = --row * res + ++col;
			} else {
				int k = i * 3;
				int triIndex = i % tris_per_row;

				int row = i / tris_per_row;
				int col = triIndex / 2;
				tris_indices[k] = row * res + ++col;
				tris_indices[k + 1] = ++row * res + --col;
				tris_indices[k + 2] = row * res + ++col;
			}
		}
	}
	// show indices
	// for (int i = 0; i < n_tris * 3; i++) {
	//     std::cout << tris_indices[i] << " ";
	//     if (i % 3 == 2) {
	//         std::cout << std::endl;
	//     }
	// }

	glGenVertexArrays(1, plane_VAO);
	glGenBuffers(1, plane_VBO);
	glGenBuffers(1, plane_EBO);

	glBindVertexArray(*plane_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, *plane_VBO);
	glBufferData(GL_ARRAY_BUFFER, size * sizeof(float), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *plane_EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, n_tris * 3 * sizeof(unsigned int), tris_indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);
	glBindVertexArray(0);
}

}// namespace ing::terrain
