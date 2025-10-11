#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_pbr::atmosphere::functions::uv_to_ray_direction

#import noisy_bevy::fbm_simplex_3d

@group(0) @binding(0) var<uniform> sim_sec: f32;

const low_sky_color = vec3(1.0, 0.7, 0.5);
const high_sky_color = vec3(0.2, 0.4, 0.7);

const sun_moon_size = 0.04;
const sun_bloom_intensity = 0.00005;
const moon_bloom_intensity = 0.00001;

const cloud_vel = vec2(0.02, 0.05);
const morph_factor = 0.05;
const cloud_height = 0.5;
const bright_cloud_brightness = 0.8;
const dark_cloud_brightness = 0.4;

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = uv_to_ray_direction(in.uv);
    let sun_dir = common::sun_dir(sim_sec);
    let moon_dir = common::moon_dir(sun_dir);

    let mapped_sun_height = common::map_sky_height(sun_dir.y);
    let mapped_moon_height = common::map_sky_height(moon_dir.y);
    let brightness = common::sky_brightness(mapped_sun_height, mapped_moon_height);

    var out = mix(
        low_sky_color,
        high_sky_color,
        common::map_sky_height(ray_dir.y),
    ) * brightness;

    let sun_dist = distance(ray_dir.xyz, sun_dir);
    var sun_intensity: f32;
    if sun_dist < sun_moon_size {
        sun_intensity = mix(1.0, 0.9, sun_dist / sun_moon_size);
    } else {
        sun_intensity = pow(sun_bloom_intensity, sun_dist);
    }
    out = mix(out, sun_color(mapped_sun_height), sun_intensity);

    // TODO: add lunar phases (affects appearance and position of the moon)
    let moon_dist = distance(ray_dir.xyz, moon_dir);
    var moon_intensity: f32;
    if moon_dist < sun_moon_size {
        moon_intensity = mix(1.0, 0.9, moon_dist / sun_moon_size);
    } else {
        moon_intensity = pow(moon_bloom_intensity, moon_dist);
    }
    out = mix(out, moon_color(mapped_moon_height), moon_intensity);

    let cloud_pos = vec2(
        ray_dir.x * cloud_height / ray_dir.y + cloud_vel.x * sim_sec,
        ray_dir.z * cloud_height / ray_dir.y + cloud_vel.y * sim_sec,
    );
    let noise = fbm_simplex_3d(vec3(cloud_pos, sim_sec * morph_factor), 4, 2.0, 0.5) / 2.0 + 0.5;
    let cloud_color = vec3(mix(bright_cloud_brightness, dark_cloud_brightness, noise) * brightness);
    let dist_scale = pow(max(ray_dir.y, 0.0), 0.2);
    out = mix(out, cloud_color, noise * dist_scale);

    return vec4(out, 1.0);
}

fn sun_color(mapped_sun_height: f32) -> vec3<f32> {
    return mix(vec3(1.0, 0.2, 0.0), vec3(1.0, 0.9, 0.8), mapped_sun_height);
}

// TODO: add texture
fn moon_color(mapped_moon_height: f32) -> vec3<f32> {
    return mix(vec3(0.5, 0.1, 0.0), vec3(0.5), mapped_moon_height);
}
