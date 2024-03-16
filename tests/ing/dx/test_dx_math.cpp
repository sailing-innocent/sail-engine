#include "test_util.h"
#include <windows.h>// for XMVerifyCPUSupport
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#include <iostream>

using namespace std;
using namespace DirectX;
using namespace DirectX::PackedVector;

bool vec4eq(XMVECTOR V, float x, float y, float z, float w) {
	return XMVector4Equal(V, XMVectorSet(x, y, z, w));
}

TEST_CASE("dx_math") {
	// XMVECTOR
	XMVECTOR v1 = XMVectorSet(1.0f, 2.0f, 3.0f, 4.0f);
	REQUIRE(vec4eq(v1, 1.0f, 2.0f, 3.0f, 4.0f));
	XMVECTOR n = XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f);
	XMVECTOR u = XMVectorSet(1.0f, 2.0f, 3.0f, 0.0f);
	XMVECTOR v = XMVectorSet(-2.0f, 1.0f, -3.0f, 0.0f);
	XMVECTOR w = XMVectorSet(0.707f, 0.707f, 0.0f, 0.0f);
	XMVECTOR a = u + v;
	REQUIRE(vec4eq(a, -1.0f, 3.0f, 0.0f, 0.0f));
	XMVECTOR b = u - v;
	REQUIRE(vec4eq(b, 0.0f, -2.0f, -3.0f, 0.0f));
	XMVECTOR c = 10.0f * u;
	REQUIRE(vec4eq(c, 10.0f, 20.0f, 30.0f, 0.0f));
	XMVECTOR L = XMVector3Length(u);
	XMVECTOR d = XMVector3Normalize(u);
	XMVECTOR s = XMVector3Dot(u, v);
	XMVECTOR e = XMVector3Cross(u, v);
}