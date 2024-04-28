struct VS_INPUT {
    @location(0) pos : vec3f,
    @location(1) color : vec3f
};

struct VS_OUTPUT {
    @builtin(position) position : vec4f,
    @location(0) color : vec4f
};

@vertex
fn main(
input : VS_INPUT
) -> VS_OUTPUT {
    var out : VS_OUTPUT;
    out.position = vec4f(input.pos, 1.0);
    out.color = vec4f(input.color, 1.0);
    return out;
}
