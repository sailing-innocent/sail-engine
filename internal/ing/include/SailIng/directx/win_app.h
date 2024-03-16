#pragma once
/**
 * @file app/win/win_app.h
 * @author sailing-innocent
 * @date 2023-03-25
 * @brief The Basic Class for Win App
 */

#include "SailIng/app.h"
#include "win32_utils.h"

namespace sail::ing {

class SAIL_ING_API INGWinApp : public INGApp {
public:
	INGWinApp(UINT width, UINT height, std::string name);
	virtual ~INGWinApp();
	// self method
	virtual void logic_tick() = 0;
	virtual void render_tick() = 0;

	// callbacks
	virtual void on_key_down(UINT8) = 0;
	virtual void on_key_up(UINT8) = 0;
	virtual void on_size_changed(UINT width, UINT height, bool minimized) = 0;

	// get-set
	UINT get_width() const { return m_width; }
	UINT get_height() const { return m_height; }
	const CHAR* get_title() const { return m_title.c_str(); }
	void set_window_bounds(int left, int top, int right, int bottom);

protected:
	UINT m_width;
	UINT m_height;
	// Adapter info.
	bool m_use_warp_device;

	// window bounds
	RECT m_window_bounds;
	// aspect ratio
	float m_aspect_ratio;

private:
	std::string m_title;
};// class INGWinApp

}// namespace sail::ing
