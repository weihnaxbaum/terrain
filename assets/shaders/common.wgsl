#define_import_path common

const grass_color: vec3<f32> = vec3(0.1, 0.4, 0.0);

// Returns the vector pointing to the sun
fn sun_dir(time: f32) -> vec3<f32> {
    const day_length = 100.0;
    return normalize(vec3(
        sin(time / day_length),
        cos(time / day_length),
        0.5,
    ));
}

// Relative to the sun's brightness
const moon_brightness: f32 = 0.3;

// TODO: improve
fn moon_dir(sun_dir: vec3<f32>) -> vec3<f32> {
    return vec3(-sun_dir.xy, sun_dir.z);
}

fn sky_brightness(mapped_sun_height: f32, mapped_moon_height: f32) -> f32 {
    return clamp(mapped_sun_height + mapped_moon_height * moon_brightness, 0.1, 1.0);
}

fn map_sky_height(ray_dir_y: f32) -> f32 {
    return pow(smoothstep(0.0, 1.0, ray_dir_y), 0.3);
}
