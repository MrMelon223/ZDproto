#version 460

uniform mat4 matVP;
uniform mat4 matGeo;

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec4 color;
layout (location = 3) in vec2 uv;

out vec3 pos_out;
out vec3 norm_out;
out vec4 color_out;
out vec2 uv_out;

void main() {
   norm_out = normal;
   color_out = color;
   uv_out = uv;
   gl_Position = matVP * matGeo * vec4(pos, 1);
   pos_out = pos;
}