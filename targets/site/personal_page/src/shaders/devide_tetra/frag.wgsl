struct VS_OUTPUT {
    @builtin(position) position : vec4f,
    @location(0) color : vec4f
};
@fragment
fn main(
v : VS_OUTPUT
) -> @location(0) vec4f {
    return v.color;
}
