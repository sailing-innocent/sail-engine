#version 450

layout(location=0)in vec4 pos;
layout(location=1)in vec4 color;

layout(binding=0)uniform UniformBufferObject
{
    mat4 model;
    mat4 view;
    mat4 proj;
}ubo;

layout(location=0)out vec4 fragColor;

void main()
{
    gl_Position=ubo.proj*ubo.view*ubo.model*pos;
    fragColor=color;
}
