// demo_curve_fit
#include "SailIng/opengl/basic_app.h"

#include <iostream>
#include <random>
#include <functional>

namespace sail::test {

class Polynomial {
public:
	Polynomial() = default;
	explicit Polynomial(int _order) {
		mOrder = _order;
		mParams.resize(_order + 1);
		for (auto i = 0; i <= mOrder; i++) {
			mParams[i] = 0.0;
		}
	}
	float forward(float x) {
		float res = 0.0;
		for (auto i = 0; i <= mOrder; i++) {
			res += std::pow(x, i) * mParams[i];
		}
		return res;
	}
	float operator()(float x) {
		return static_cast<float>(forward(static_cast<float>(x)));
	}
	bool setParam(int _order, float _param) {
		mParams[_order] = _param;
		return true;
	}
	friend std::ostream& operator<<(std::ostream& os, Polynomial poly) {
		os << poly.params()[0];
		for (auto i = 1; i <= poly.order(); i++) {
			os << "+" << poly.params()[i] << "x^" << i;
		}
		os << std::endl;
		return os;
	}
	const int order() const { return mOrder; }
	std::vector<float>& params() { return mParams; }

	std::vector<float> ESM_dir(std::vector<float>& samples_x, std::vector<float>& samples_y, int samples_N) {
		std::vector<float> res;
		for (auto n = 0; n <= mOrder; n++) {
			float dEdw = 0.0;
			for (auto j = 0; j < samples_N; j++) {
				dEdw += (forward(samples_x[j]) - samples_y[j]) * std::pow(samples_x[j], n);
			}
			res.push_back(dEdw);
			// std::cout << dEdw << ",";
		}
		// std::cout << std::endl;
		return res;
	}

protected:
	int mOrder;
	std::vector<float> mParams;
};

float disp(float d, float d_min, float d_max, float display_min = -0.8f, float display_max = 0.8f) {
	return (d - d_min) * (display_max - display_min) / (d_max - d_min) + display_min;
}

class PolynomialFitApp {
public:
	PolynomialFitApp() = default;
	~PolynomialFitApp() { m_app.terminate(); }
	void init() {
		draw_axis();
		m_app.init();
	}
	void draw_points(ing::GLPointList points) { m_app.addPoints(points); }

	void draw_fn(std::function<float(float)> fn,
				 std::vector<float> color = {0.0f, 1.0f, 0.0f, 1.0f}) {
		size_t N = 100;
		float start = x_min;
		float end = x_max;
		float gap = (end - start) / (N - 1);
		std::vector<float> data;
		for (auto i = 0; i < N; i++) {
			data.push_back(start + i * gap);
		}

		float y_disp = disp(fn(data[0]), y_min, y_max);
		float x_disp = disp(data[0], x_min, x_max);
		ing::GLPoint startPoint(x_disp, y_disp);
		startPoint.setColor(color);
		ing::GLPoint prevPoint = startPoint;
		for (auto i = 1; i < N; i++) {
			y_disp = disp(fn(data[i]), y_min, y_max);
			x_disp = disp(data[i], x_min, x_max);
			ing::GLPoint point(x_disp, y_disp);
			point.setColor(color);
			ing::GLLine line(prevPoint, point);
			// std::cout << point.vertices()[0] << "," << data[i] << std::endl;
			m_app.addLine(line);
			prevPoint = point;
		}
	}
	void draw_axis() {
		std::vector<float> blue{0.0f, 0.0f, 1.0f, 1.0f};
		ing::GLPoint x_left(x_min);
		x_left.setColor(blue);
		ing::GLPoint x_right(x_max);
		x_right.setColor(blue);
		ing::GLPoint y_top(0.0f, y_max);
		y_top.setColor(blue);
		ing::GLPoint y_buttom(0.0f, y_min);
		y_buttom.setColor(blue);
		ing::GLLine x_axis(x_left, x_right);
		ing::GLLine y_axis(y_buttom, y_top);
		m_app.addLine(x_axis);
		m_app.addLine(y_axis);
	}

	void show() {
		while (m_app.tick()) {
		}
	}

private:
	ing::INGGLBasicApp m_app;
	float x_min = -1.0f;
	float x_max = 1.0f;
	float y_min = -1.0f;
	float y_max = 1.0f;
};

}// namespace sail::test

using namespace sail;
using namespace sail::test;

int main() {
	PolynomialFitApp app{};
	std::function<float(float)> fn = [](float x) {
		const float PI = 3.1415926f;
		return std::sinf(2 * PI * x);
	};
	app.draw_fn(fn);

	// sampling
	std::default_random_engine e;
	std::uniform_real_distribution<float> u(0.0f, 1.0f);
	size_t sample_N = 20;
	float sample_x_min = 0.0f;
	float sample_x_max = 1.0f;
	std::vector<float> samples_x;
	std::vector<float> samples_y;
	std::vector<float> yellow = {1.0f, 1.0f, 0.0f, 1.0f};
	ing::GLPointList sample_points;
	for (auto i = 0; i < sample_N; i++) {
		float sample_x = sample_x_min + (sample_x_max - sample_x_min) * u(e);
		float sample_y = fn(sample_x) + u(e) * 0.2 - 0.1;
		samples_x.push_back(sample_x);
		samples_y.push_back(sample_y);
		float disp_x = disp(sample_x, -1.0, 1.0);
		float disp_y = disp(sample_y, -1.0, 1.0);
		ing::GLPoint sample_p(disp_x, disp_y);

		sample_p.setColor(yellow);
		// std::cout << sample_p.vertices()[0] << "," << sample_p.vertices()[1] << std::endl;
		sample_points.appendPrimitive(sample_p);
	}
	app.draw_points(sample_points);
	const int M = 5;
	Polynomial poly(M);

	const int nsteps = 10000;
	const float alpha = 0.01f;

	for (auto s = 0; s < nsteps; s++) {

		std::vector<float> derivate = poly.ESM_dir(samples_x, samples_y, sample_N);
		for (auto i = 0; i <= M; i++) {
			poly.setParam(i, poly.params()[i] - alpha * derivate[i]);
		}
		if (s % 100 == 0) {
			std::vector<float> fcolor = {0.0f, 0.0f, 1.0f / nsteps * s, 1.0f};
			app.draw_fn(poly, fcolor);
		}
	}
	// app.draw()
	// app.fit(samples)
	app.init();
	app.show();

	return 0;
}
