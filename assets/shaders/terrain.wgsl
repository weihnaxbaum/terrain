#import bevy_pbr::{
    forward_io::Vertex,
    mesh_functions,
    mesh_view_bindings::{globals, view},
    view_transformations::position_world_to_clip
}
#import noisy_bevy::simplex_noise_2d

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sim_sec: f32;

const fog_density = 0.0002;
const fog_color = vec3(0.6, 0.6, 0.8);

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) slope: vec2<f32>,
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;

    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    out.world_pos = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(in.position, 1.0)).xyz;
    let noise = noise(out.world_pos.xz);
    out.world_pos.y = noise.x;
    out.slope = noise.yz;
    out.clip_pos = position_world_to_clip(out.world_pos.xyz);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(vec3(-in.slope.x, 1.0, -in.slope.y));
    let sun_dir = common::sun_dir(sim_sec);
    let moon_dir = common::moon_dir(sun_dir);
    let sun_height = common::map_sky_height(sun_dir.y);
    let moon_height = common::map_sky_height(moon_dir.y);
    let sky_brightness = common::sky_brightness(sun_height, moon_height);

    let brightness = clamp(
        max(dot(normal, sun_dir) * sun_height, 0.0) +
        max(dot(normal, moon_dir) * moon_height * common::moon_brightness, 0.0),
        0.1,
        1.0,
    );
    let slope = clamp(length(in.slope * 0.5), 0.0, 1.0);
    let albedo = mix(common::grass_color, vec3(0.2, 0.2, 0.1), slope);
    var out = albedo * brightness;
    let fog = exp(-fog_density * distance(in.world_pos, view.world_position));
    out = mix(fog_color * sky_brightness, out, fog);
    return vec4(out, 1.0);
}

// Returns:
// - x: height
// - yz: slope
fn noise(pos: vec2<f32>) -> vec3<f32> {
    // Update `TERRAIN_MIN_HEIGHT` and `TERRAIN_MAX_HEIGHT` in Rust code
    // when changing these values

    // https://youtu.be/gsJHzBTPG0Y
    const slope_amp_falloff = 10.0;

    var freq = 0.005;
    var amp = 1.0;

    var height = 0.5;
    var slope = vec2(0.0);

    for (var octave = 0; octave < 9; octave++) {
        let y = simplex_noise_2d(pos * freq);
        // TODO: calculate using the derivative
        slope += vec2(
            simplex_noise_2d(vec2(pos.x + 0.01, pos.y) * freq) - y,
            simplex_noise_2d(vec2(pos.x, pos.y + 0.01) * freq) - y,
        ) / 0.01 * amp;
        height += y * amp / (1.0 + slope_amp_falloff * length(slope));
        freq *= 2.0;
        amp *= 0.5;
    }
    return vec3(transform_height(height), slope * transform_height_derivative(height));
}

fn transform_height(height: f32) -> f32 {
    return mix(height, height * height, sign(height) * 0.5 + 0.5) * 15.0;
}

fn transform_height_derivative(height: f32) -> f32 {
    // Power rule
    return mix(1.0, 2.0 * height, sign(height) * 0.5 + 0.5) * 15.0;
}
