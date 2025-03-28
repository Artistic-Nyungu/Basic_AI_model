#version 460 core

layout(location = 0) in vec2 aPos;
out vec2 uv;

vec2 transform = vec2(1.f, -1.f);

void main() {
    vec2 pos = aPos * transform;
    gl_Position = vec4(pos, 1.0f, 1.0f);
    uv = pos * 0.5f + 0.5f;
}