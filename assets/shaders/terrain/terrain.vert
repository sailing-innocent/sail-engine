#version 450

layout(location = 0) in vec3 position_vs_in;
layout(location = 1) in vec3 normal_vs_in;
layout(location = 2) in vec2 texcoord_vs_in;
layout(location = 3) in vec2 a_position;

uniform mat4 g_world;
uniform mat4 g_vp;

// out vec3 world_pos_cs_in;
// out vec2 texcoord_pos_cs_in;
// out vec3 normal_cs_in;

void main()
{
    // world_pos_cs_in = (g_world * vec4(position_vs_in, 1.0f)).xyz;
    // world_pos_cs_in.xz += a_position;
    // normal_cs_in       = normal_vs_in;
    // texcoord_pos_cs_in = texcoord_vs_in;
    float x = position_vs_in.x / 1000.0f;
    float z = position_vs_in.z / 1000.0f;
    float y = x * x + z * z;
    gl_Position = vec4(x, y, z, 1.0f);
    // gl_Position = vec4(position_vs_in.x / 1000.f, position_vs_in.z / 1000.f, 0.0f , 1.0f);
}
