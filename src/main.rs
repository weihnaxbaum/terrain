#![cfg_attr(bevy_lint, feature(register_tool), register_tool(bevy))]
#![cfg_attr(not(feature = "console"), windows_subsystem = "windows")]

use std::{array, borrow::Cow, f32::consts::FRAC_PI_2, mem, result::Result, time::Duration};

use bevy::{
    camera::primitives::Aabb,
    core_pipeline::{
        FullscreenShader,
        core_3d::graph::{Core3d, Node3d},
    },
    ecs::{query::QueryItem, system::lifetimeless::Read},
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    mesh::PlaneMeshBuilder,
    platform::collections::HashSet,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        globals::{GlobalsBuffer, GlobalsUniform},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            AsBindGroup, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, MultisampleState,
            PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, Sampler, SamplerBindingType, ShaderStages,
            SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat,
            TextureSampleType, TextureUsages, VertexState,
            binding_types::{sampler, texture_2d, texture_2d_multisampled, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
    shader::ShaderRef,
    time::common_conditions::on_timer,
    window::WindowMode,
};
use noisy_bevy::NoisyShaderPlugin;

fn main() -> AppExit {
    App::new()
        .add_plugins((
            DefaultPlugins,
            NoisyShaderPlugin,
            MaterialPlugin::<TerrainMaterial>::default(),
            #[cfg(feature = "frame_time_diagnostics")]
            (
                bevy::diagnostic::LogDiagnosticsPlugin::default(),
                bevy::diagnostic::FrameTimeDiagnosticsPlugin::default(),
            ),
            SkyPlugin,
            WaterPlugin,
        ))
        .init_state::<AppState>()
        .add_systems(Startup, (setup, update_chunks).chain())
        .add_systems(
            Update,
            (
                (
                    update_chunks.run_if(on_timer(Duration::from_secs(1))),
                    move_cam,
                )
                    .run_if(in_state(AppState::Running)),
                update_state,
                toggle_fullscreen,
            ),
        )
        .add_systems(OnEnter(AppState::Paused), on_pause)
        .run()
}

#[derive(States, Debug, PartialEq, Eq, Hash, Clone, Default)]
#[states(scoped_entities)]
enum AppState {
    #[default]
    Running,
    Paused,
}

#[derive(AsBindGroup, Clone, Asset, TypePath)]
struct TerrainMaterial {}

impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/terrain.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/terrain.wgsl".into()
    }
}

/// (start_height - (1 - gain ^ (octaves + 1)) / (1 - gain)) ^ 2 * amp
const TERRAIN_MIN_HEIGHT: f32 = -22.470703;
/// (start_height + (1 - gain ^ (octaves + 1)) / (1 - gain)) ^ 2 * amp
const TERRAIN_MAX_HEIGHT: f32 = 93.60358;

const TERRAIN_CENTER_HEIGHT: f32 = (TERRAIN_MIN_HEIGHT + TERRAIN_MAX_HEIGHT) / 2.0;
const TERRAIN_HALF_HEIGHT: f32 = TERRAIN_MAX_HEIGHT - TERRAIN_CENTER_HEIGHT;

const LOD_COUNT: u8 = 6;

#[derive(Component)]
struct Chunk;

#[derive(Resource)]
struct ChunkMeshes([Handle<Mesh>; LOD_COUNT as usize]);

impl ChunkMeshes {
    fn get(&self, dist_squared: f32) -> Handle<Mesh> {
        let i = if dist_squared < 40000.0 {
            0
        } else if dist_squared < 160000.0 {
            1
        } else if dist_squared < 640000.0 {
            2
        } else if dist_squared < 2560000.0 {
            3
        } else if dist_squared < 10240000.0 {
            4
        } else {
            5
        };
        self.0[i].clone()
    }
}

#[derive(Resource)]
struct TerrainMaterialHandle(Handle<TerrainMaterial>);

const CHUNK_SIZE: f32 = 200.0;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
) {
    commands.spawn(Camera3d {
        depth_texture_usages: (TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING)
            .into(),
        ..default()
    });
    mem::forget(asset_server.load::<Shader>("shaders/common.wgsl"));

    commands.insert_resource(ChunkMeshes(array::from_fn(|i| {
        meshes.add(
            PlaneMeshBuilder {
                plane: Plane3d {
                    half_size: Vec2::splat(CHUNK_SIZE / 2.0),
                    ..default()
                },
                subdivisions: 1024 / 2u32.pow(i as u32),
            }
            .build(),
        )
    })));
    commands.insert_resource(TerrainMaterialHandle(materials.add(TerrainMaterial {})));
}

fn update_state(
    state: Res<State<AppState>>,
    mut next: ResMut<NextState<AppState>>,
    kb: Res<ButtonInput<KeyCode>>,
) {
    if kb.just_pressed(KeyCode::Escape) {
        next.set(match *state.get() {
            AppState::Running => AppState::Paused,
            AppState::Paused => AppState::Running,
        });
    }
}

fn on_pause(mut commands: Commands) {
    commands.spawn((
        DespawnOnExit(AppState::Paused),
        Node {
            width: Val::Percent(90.0),
            height: Val::Percent(90.0),
            border: UiRect::all(Val::Percent(0.5)),
            flex_direction: FlexDirection::Column,
            justify_self: JustifySelf::Center,
            align_self: AlignSelf::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
        BorderRadius::all(Val::Percent(5.0)),
        BorderColor::all(Color::BLACK),
        children![
            (Text::new("Paused"), TextFont::from_font_size(50.0)),
            (
                Button,
                Node {
                    height: Val::Px(50.0),
                    width: Val::Percent(70.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BorderRadius::all(Val::Percent(100.0)),
                children![(
                    Text::new("Toggle fullscreen"),
                    TextFont::from_font_size(30.0),
                )]
            )
        ],
    ));
}

fn toggle_fullscreen(
    mut q: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<Button>)>,
    mut window: Single<&mut Window>,
) {
    for (interaction, mut bg) in &mut q {
        bg.0 = match *interaction {
            Interaction::Pressed => {
                window.mode = match window.mode {
                    WindowMode::Windowed => {
                        WindowMode::BorderlessFullscreen(MonitorSelection::Current)
                    }
                    _ => WindowMode::Windowed,
                };
                Color::srgb(0.4, 0.4, 0.4)
            }
            Interaction::Hovered => Color::srgb(0.2, 0.2, 0.2),
            Interaction::None => Color::BLACK,
        };
    }
}

const RENDER_DIST: i32 = 32;

fn update_chunks(
    chunk_q: Query<(&Transform, Entity), With<Chunk>>,
    mut commands: Commands,
    meshes: Res<ChunkMeshes>,
    material: Res<TerrainMaterialHandle>,
    cam: Single<&Transform, With<Camera>>,
) {
    let mut chunks = HashSet::new();
    for (tf, e) in chunk_q.iter() {
        let dist_squared = tf.translation.xz().distance_squared(cam.translation.xz());
        if dist_squared > (RENDER_DIST * RENDER_DIST) as f32 * CHUNK_SIZE * CHUNK_SIZE {
            commands.entity(e).despawn();
            continue;
        }
        chunks.insert((tf.translation.xz() / CHUNK_SIZE).as_ivec2());
        commands.entity(e).insert(Mesh3d(meshes.get(dist_squared)));
    }
    for z in -RENDER_DIST..RENDER_DIST {
        for x in -RENDER_DIST..RENDER_DIST {
            let pos = Vec2::new(x as f32, z as f32);
            if pos.length_squared() > (RENDER_DIST * RENDER_DIST) as f32 {
                continue;
            }
            let pos = pos + (cam.translation.xz() / CHUNK_SIZE).round();
            if chunks.contains(&pos.as_ivec2()) {
                continue;
            }
            let pos = pos * CHUNK_SIZE;
            commands.spawn((
                Chunk,
                Mesh3d(meshes.get(pos.distance_squared(cam.translation.xz()))),
                MeshMaterial3d(material.0.clone()),
                Transform::from_xyz(pos.x, 0.0, pos.y),
                Aabb {
                    center: Vec3A::new(0.0, TERRAIN_CENTER_HEIGHT, 0.0),
                    half_extents: Vec3A::new(
                        CHUNK_SIZE / 2.0,
                        TERRAIN_HALF_HEIGHT,
                        CHUNK_SIZE / 2.0,
                    ),
                },
            ));
        }
    }
}

fn move_cam(
    accumulated_mouse_motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    kb: Res<ButtonInput<KeyCode>>,
    mut tf: Single<&mut Transform, With<Camera>>,
    mut speed: Local<f32>,
    time: Res<Time>,
) {
    let sensi = Vec2::new(0.003, 0.002);

    let delta = accumulated_mouse_motion.delta;

    if delta != Vec2::ZERO {
        // Note that we are not multiplying by delta_time here.
        // The reason is that for mouse movement, we already get the full movement that happened since the last frame.
        // This means that if we multiply by delta_time, we will get a smaller rotation than intended by the user.
        // This situation is reversed when reading e.g. analog input from a gamepad however, where the same rules
        // as for keyboard input apply. Such an input should be multiplied by delta_time to get the intended rotation
        // independent of the framerate.
        let delta_yaw = -delta.x * sensi.x;
        let delta_pitch = -delta.y * sensi.y;

        let (yaw, pitch, roll) = tf.rotation.to_euler(EulerRot::YXZ);
        let yaw = yaw + delta_yaw;

        // If the pitch was ±¹⁄₂ π, the camera would look straight up or down.
        // When the user wants to move the camera back to the horizon, which way should the camera face?
        // The camera has no way of knowing what direction was "forward" before landing in that extreme position,
        // so the direction picked will for all intents and purposes be arbitrary.
        // Another issue is that for mathematical reasons, the yaw will effectively be flipped when the pitch is at the extremes.
        // To not run into these issues, we clamp the pitch to a safe range.
        const PITCH_LIMIT: f32 = FRAC_PI_2 - 0.01;
        let pitch = (pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);

        tf.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }

    if *speed == 0.0 {
        *speed = 50.0;
    }
    *speed += match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y * 5.0,
        MouseScrollUnit::Pixel => scroll.delta.y * 0.25,
    };
    *speed = speed.max(1.0);

    let mut dir = Vec3::ZERO;

    if kb.pressed(KeyCode::KeyW) {
        dir.z -= 1.0;
    }
    if kb.pressed(KeyCode::KeyS) {
        dir.z += 1.0;
    }
    if kb.pressed(KeyCode::KeyA) {
        dir.x -= 1.0;
    }
    if kb.pressed(KeyCode::KeyD) {
        dir.x += 1.0;
    }
    if kb.pressed(KeyCode::ShiftLeft) {
        dir.y -= 1.0;
    }
    if kb.pressed(KeyCode::Space) {
        dir.y += 1.0;
    }

    let rot = tf.rotation;
    tf.translation += rot * dir.normalize_or_zero() * *speed * time.delta_secs();
}

struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        app.get_sub_app_mut(RenderApp)
            .expect("No RenderApp")
            .init_resource::<SkyPipelineSpecializer>()
            .init_resource::<SpecializedRenderPipelines<SkyPipelineSpecializer>>()
            .add_systems(
                Render,
                (
                    queue_sky_pipeline.in_set(RenderSystems::Queue),
                    prepare_sky_bind_group.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<RenderSkyNode>>(Core3d, RenderSkyLabel)
            .add_render_graph_edge(Core3d, RenderSkyLabel, Node3d::MainOpaquePass);
    }
}

#[derive(Resource)]
struct SkyPipelineSpecializer {
    frag: Handle<Shader>,
    vert: VertexState,
    layout: BindGroupLayout,
}

impl FromWorld for SkyPipelineSpecializer {
    fn from_world(world: &mut World) -> Self {
        let rd = world.resource::<RenderDevice>();
        Self {
            frag: world.load_asset("shaders/sky.wgsl"),
            vert: world.resource::<FullscreenShader>().to_vertex_state(),
            layout: rd.create_bind_group_layout(
                "sky_bind_group_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::FRAGMENT,
                    // Bevy's atmosphere shader functions assume a specific
                    // [layout](https://github.com/bevyengine/bevy/blob/main/crates/bevy_pbr/src/atmosphere/bindings.wgsl).
                    // Bevy's mesh functions assume a different
                    // [layout](https://github.com/bevyengine/bevy/blob/main/crates/bevy_pbr/src/render/mesh_view_bindings.wgsl).
                    // Here we just mix 'n match to make it work :)
                    (
                        (3, uniform_buffer::<ViewUniform>(true)),
                        (11, uniform_buffer::<GlobalsUniform>(false)),
                    ),
                ),
            ),
        }
    }
}

impl SpecializedRenderPipeline for SkyPipelineSpecializer {
    type Key = PipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: None,
            layout: vec![self.layout.clone()],
            push_constant_ranges: vec![],
            vertex: self.vert.clone(),
            primitive: default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples,
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.frag.clone(),
                shader_defs: vec![],
                entry_point: Some(Cow::Borrowed("main")),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: None,
                    write_mask: ColorWrites::COLOR,
                })],
            }),
            zero_initialize_workgroup_memory: true,
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct PipelineKey {
    msaa_samples: u32,
}

#[derive(Component)]
struct SkyPipelineId(CachedRenderPipelineId);

fn queue_sky_pipeline(
    cam: Single<(Entity, &Msaa), With<Camera>>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<SkyPipelineSpecializer>,
    mut specializer: ResMut<SpecializedRenderPipelines<SkyPipelineSpecializer>>,
    mut commands: Commands,
) {
    let id = specializer.specialize(
        &pipeline_cache,
        &layouts,
        PipelineKey {
            msaa_samples: cam.1.samples(),
        },
    );
    commands.entity(cam.0).insert(SkyPipelineId(id));
}

#[derive(Component)]
struct SkyBindGroup(BindGroup);

fn prepare_sky_bind_group(
    cam: Single<Entity, With<Camera>>,
    rd: Res<RenderDevice>,
    specializer: Res<SkyPipelineSpecializer>,
    view_uniforms: Res<ViewUniforms>,
    globals_buffer: Res<GlobalsBuffer>,
    mut commands: Commands,
) {
    let view_bindings = view_uniforms
        .uniforms
        .binding()
        .expect("Could not create view bindings for sky bind group");
    let globals_binding = globals_buffer
        .buffer
        .binding()
        .expect("Could not create globals bindings for sky bind group");
    let bind_group = rd.create_bind_group(
        "sky_bind_group",
        &specializer.layout,
        &BindGroupEntries::with_indices(((3, view_bindings), (11, globals_binding))),
    );
    commands.entity(*cam).insert(SkyBindGroup(bind_group));
}

#[derive(RenderLabel, Hash, Debug, PartialEq, Eq, Clone)]
struct RenderSkyLabel;

#[derive(Default)]
struct RenderSkyNode;

impl ViewNode for RenderSkyNode {
    type ViewQuery = (
        Read<SkyPipelineId>,
        Read<ViewTarget>,
        Read<SkyBindGroup>,
        Read<ViewUniformOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (pipeline_id, view_target, bind_group, view_uniform_offset): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
            return Ok(());
        };
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(view_target.get_color_attachment())],
                ..default()
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group.0, &[view_uniform_offset.offset]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        app.get_sub_app_mut(RenderApp)
            .expect("No RenderApp")
            .init_resource::<WaterPipelineSpecializer>()
            .init_resource::<SpecializedRenderPipelines<WaterPipelineSpecializer>>()
            .add_systems(Render, queue_water_pipeline.in_set(RenderSystems::Queue))
            .add_render_graph_node::<ViewNodeRunner<RenderWaterNode>>(Core3d, RenderWaterLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::Tonemapping,
                    RenderWaterLabel,
                    Node3d::EndMainPassPostProcessing,
                ),
            );
    }
}

#[derive(Resource)]
struct WaterPipelineSpecializer {
    frag: Handle<Shader>,
    vert: VertexState,
    layout: BindGroupLayout,
    sampler: Sampler,
}

impl FromWorld for WaterPipelineSpecializer {
    fn from_world(world: &mut World) -> Self {
        let rd = world.resource::<RenderDevice>();
        Self {
            frag: world.load_asset("shaders/water.wgsl"),
            vert: world.resource::<FullscreenShader>().to_vertex_state(),
            layout: rd.create_bind_group_layout(
                "water_bind_group_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::FRAGMENT,
                    (
                        (0, texture_2d_multisampled(TextureSampleType::Depth)),
                        (
                            1,
                            texture_2d(TextureSampleType::Float { filterable: false }),
                        ),
                        (2, sampler(SamplerBindingType::NonFiltering)),
                        (3, uniform_buffer::<ViewUniform>(true)),
                        (11, uniform_buffer::<GlobalsUniform>(false)),
                    ),
                ),
            ),
            sampler: rd.create_sampler(&default()),
        }
    }
}

impl SpecializedRenderPipeline for WaterPipelineSpecializer {
    type Key = ();

    fn specialize(&self, _key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: None,
            layout: vec![self.layout.clone()],
            push_constant_ranges: vec![],
            vertex: self.vert.clone(),
            primitive: default(),
            depth_stencil: None,
            multisample: default(),
            fragment: Some(FragmentState {
                shader: self.frag.clone(),
                shader_defs: vec![],
                entry_point: Some(Cow::Borrowed("main")),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: None,
                    write_mask: ColorWrites::COLOR,
                })],
            }),
            zero_initialize_workgroup_memory: true,
        }
    }
}

#[derive(Component)]
struct WaterPipelineId(CachedRenderPipelineId);

fn queue_water_pipeline(
    cam: Single<(Entity, &Msaa), With<Camera>>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<WaterPipelineSpecializer>,
    mut specializer: ResMut<SpecializedRenderPipelines<WaterPipelineSpecializer>>,
    mut commands: Commands,
) {
    let id = specializer.specialize(&pipeline_cache, &layouts, ());
    commands.entity(cam.0).insert(WaterPipelineId(id));
}

#[derive(RenderLabel, Hash, Debug, PartialEq, Eq, Clone)]
struct RenderWaterLabel;

#[derive(Default)]
struct RenderWaterNode;

impl ViewNode for RenderWaterNode {
    type ViewQuery = (
        Read<WaterPipelineId>,
        Read<ViewTarget>,
        Read<ViewDepthTexture>,
        Read<ViewUniformOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (pipeline_id, view_target, view_depth_texture, view_uniform_offset): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
            return Ok(());
        };

        let pipeline_specializer = world.resource::<WaterPipelineSpecializer>();
        let post_process = view_target.post_process_write();
        let view_bindings = world
            .resource::<ViewUniforms>()
            .uniforms
            .binding()
            .expect("Could not create view bindings for water bind group");
        let globals_binding = world
            .resource::<GlobalsBuffer>()
            .buffer
            .binding()
            .expect("Could not create globals bindings for water bind group");
        let bind_group = render_context.render_device().create_bind_group(
            "water_bind_group",
            &pipeline_specializer.layout,
            &BindGroupEntries::with_indices((
                (0, view_depth_texture.view()),
                (1, post_process.source),
                (2, &pipeline_specializer.sampler),
                (3, view_bindings),
                (11, globals_binding),
            )),
        );

        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: post_process.destination,
                    depth_slice: None,
                    resolve_target: None,
                    ops: default(),
                })],
                ..default()
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
