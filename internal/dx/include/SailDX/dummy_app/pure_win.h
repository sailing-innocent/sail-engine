#pragma once
/**
 * @file app/win/pure.h
 * @author sailing-innocent
 * @date 2023-05-02
 * @brief The Basic Pure Win App
 */

#include "win_app.h"

namespace sail::dx {

class SAIL_DX_API DXPureWinApp : public DXWinApp {
public:
	DXPureWinApp(UINT width, UINT height, std::string name);
	virtual ~DXPureWinApp();
	// life cycle
	void init() override;
	bool tick(int count) override;
	void terminate() override;
	void logic_tick() override;
	void render_tick() override;
	// call backs
	void on_key_down(UINT8) override;
	void on_key_up(UINT8) override{};
	void on_size_changed(UINT width, UINT height, bool minimized) override{};
};

}// namespace sail::dx