struct VS_INPUT {
    @location(0) pos : vec3f,
};

@vertex
fn main(
input : VS_INPUT
) -> @builtin(position) vec4f {
    return vec4f(input.pos, 1.0);
}
