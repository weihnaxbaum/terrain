#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_pbr::{
    atmosphere::{
        bindings::view,
        functions::{
            ndc_to_uv,
            uv_to_ndc,
            uv_to_ray_direction,
        }
    },
    mesh_view_bindings::globals,
}

@group(0) @binding(0) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;
@group(0) @binding(4) var<uniform> sim_sec: f32;

const see_dist = 100.0;
const falloff = 0.5;
const color = vec3(0.0, 0.2, 0.6);

const reflection_color = (color + common::grass_color) * 0.5;
const reflection_ray_len = 10.0; // keep below see_dist for underwater reflections
const reflection_blend_size = 0.1;

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let in_color = textureSample(texture, texture_sampler, in.uv);

    let ray_dir = uv_to_ray_direction(in.uv);

    let depth = textureLoad(depth_texture, vec2<i32>(in.position.xy), 0);
    let cam_terrain_dist = ndc_to_camera_dist(vec3(uv_to_ndc(in.uv), depth));

    let pos = view.world_position;
    let terrain_pos = pos + ray_dir.xyz * cam_terrain_dist;
    let surface_dist = -pos.y / ray_dir.y;
    let surface_pos = pos + ray_dir.xyz * surface_dist;

    var water_depth: f32;
    if terrain_pos.y < 0.0 && pos.y <= 0.0 {
        // below water, looking at a point below water
        water_depth = distance(pos, terrain_pos);
    } else if terrain_pos.y < 0.0 && pos.y > 0.0 {
        // above water, looking at a point below water
        water_depth = distance(surface_pos, terrain_pos);
    } else if terrain_pos.y >= 0.0 && pos.y < 0.0 {
        // below water, looking at a point above water
        water_depth = distance(pos, surface_pos);
    } else {
        // above water, looking at a point above water
        water_depth = 0.0;
    }

    let intensity = water_depth_to_intensity(water_depth);
    let brightness = brightness();
    let with_water_color = mix(in_color.rgb, color * brightness, intensity);

    if surface_dist <= 0.0 || surface_dist >= cam_terrain_dist {
        return vec4(with_water_color, 1.0);
    }

    let normal = normal(surface_pos.xz, pos.y > 0.0);

    // Schlick's approximation
    const r0 = 0.04;
    let fresnel = r0 + (1.0 - r0) * clamp(pow(1.0 - clamp(dot(normal, -ray_dir.xyz), 0.0, 1.0), 5.0), 0.0, 1.0);

    let reflection = reflection(surface_pos, normal, ray_dir.xyz, brightness);
    let with_reflection = mix(with_water_color, reflection, fresnel);

    return vec4(with_reflection, 1.0);
}

fn brightness() -> f32 {
    let sun_dir = common::sun_dir(sim_sec);
    let moon_dir = common::moon_dir(sun_dir);
    let sun_height = common::map_sky_height(sun_dir.y);
    let moon_height = common::map_sky_height(moon_dir.y);
    return common::sky_brightness(sun_height, moon_height);
}

const wave_octaves = 5;
const speed = 1.0;

fn normal(pos: vec2<f32>, from_above: bool) -> vec3<f32> {
    var sum = vec2(0.0);
    var freq = 0.1;
    var amp = 0.5;
    var angle = 0.0;
    for (var i = 0; i < wave_octaves; i++) {
        let dir = vec2(cos(angle), sin(angle));
        sum += cos(sim_sec * speed + dot(pos, dir) * freq) * amp * dir * freq;
        freq *= 2.0;
        amp *= 0.5;
        angle += 1.0;
    }
    let result = normalize(vec3(sum.x, 1.0, sum.y));
    if from_above {
        return result;
    } else {
        return -result;
    }
}

fn ndc_to_camera_dist(ndc: vec3<f32>) -> f32 {
    let view_pos = view.view_from_clip * vec4(ndc, 1.0);
    let t = length(view_pos.xyz / view_pos.w);
    return t;
}

fn reflection(surface_pos: vec3<f32>, normal: vec3<f32>, ray_dir: vec3<f32>, brightness: f32) -> vec3<f32> {
    let reflect_dir = 2.0 * dot(normal, -ray_dir) * normal + ray_dir;

    let ray_pos = surface_pos + reflection_ray_len * reflect_dir;
    let clip = view.clip_from_world * vec4(ray_pos, 1.0);
    let ndc = clip.xy / clip.w;
    let uv = ndc_to_uv(ndc);

    let base_color = reflection_color * brightness;

    if uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0 {
        return base_color;
    }

    var reflection = textureSample(texture, texture_sampler, uv).xyz;
    if view.world_position.y < 0.0 {
        reflection = mix(reflection, color, water_depth_to_intensity(reflection_ray_len));
    }
    let dist_to_edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let blend = min(dist_to_edge / reflection_blend_size, 1.0);

    return mix(base_color, reflection, blend);
}

fn water_depth_to_intensity(depth: f32) -> f32 {
    return min(pow(depth / see_dist, falloff), 1.0);
}
