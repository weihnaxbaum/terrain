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
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            AsBindGroup, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, MultisampleState,
            PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, Sampler, SamplerBindingType, ShaderStages, ShaderType,
            SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat,
            TextureSampleType, TextureUsages, UniformBuffer, VertexState,
            binding_types::{sampler, texture_2d, texture_2d_multisampled, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
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
            ExtractResourcePlugin::<SimTime>::default(),
            #[cfg(feature = "frame_time_diagnostics")]
            (
                bevy::diagnostic::LogDiagnosticsPlugin::default(),
                bevy::diagnostic::FrameTimeDiagnosticsPlugin::default(),
            ),
            sky_plugin,
            water_plugin,
        ))
        .init_state::<AppState>()
        .add_systems(Startup, (setup, update_chunks).chain())
        .add_systems(
            Update,
            (
                (
                    update_chunks.run_if(on_timer(Duration::from_secs(1))),
                    move_cam,
                    tick_sim_time,
                )
                    .run_if(in_state(AppState::Running)),
                update_state,
                toggle_fullscreen,
                go_past,
                go_future,
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
struct TerrainMaterial {
    // TODO: consider using a specialized mesh pipeline
    #[uniform(0)]
    sim_sec: f32,
}

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
    commands.init_resource::<SimTime>();

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
    commands.insert_resource(TerrainMaterialHandle(
        materials.add(TerrainMaterial { sim_sec: 0.0 }),
    ));
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

#[derive(ExtractResource, Resource, ShaderType, Clone, Default)]
struct SimTime {
    sec: f32,
}

fn tick_sim_time(
    mut sim: ResMut<SimTime>,
    time: Res<Time>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
) {
    sim.sec += time.delta_secs();
    for (_id, material) in materials.iter_mut() {
        material.sim_sec = sim.sec;
    }
}

#[derive(Component)]
struct FullscreenBtn;

#[derive(Component)]
struct PastBtn;

impl PastBtn {
    const SECS: f32 = 10.0;
}

#[derive(Component)]
struct FutureBtn;

impl FutureBtn {
    const SECS: f32 = 10.0;
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
                FullscreenBtn,
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
            ),
            (
                Node {
                    height: px(50),
                    width: percent(70),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                },
                children![
                    (Text::new("Time"), TextFont::from_font_size(30.0)),
                    (
                        PastBtn,
                        Button,
                        BorderRadius::all(percent(100)),
                        children![(
                            Text(format!(" -{}s ", PastBtn::SECS)),
                            TextFont::from_font_size(30.0),
                        )]
                    ),
                    (
                        FutureBtn,
                        Button,
                        BorderRadius::all(percent(100)),
                        children![(
                            Text(format!(" +{}s ", FutureBtn::SECS)),
                            TextFont::from_font_size(30.0),
                        )]
                    ),
                ]
            )
        ],
    ));
}

fn toggle_fullscreen(
    mut q: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<FullscreenBtn>)>,
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

fn go_past(
    mut q: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<PastBtn>)>,
    mut sim_time: ResMut<SimTime>,
) {
    for (interaction, mut bg) in &mut q {
        bg.0 = match *interaction {
            Interaction::Pressed => {
                sim_time.sec -= PastBtn::SECS;
                Color::srgb(0.4, 0.4, 0.4)
            }
            Interaction::Hovered => Color::srgb(0.2, 0.2, 0.2),
            Interaction::None => Color::BLACK,
        };
    }
}

fn go_future(
    mut q: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<FutureBtn>)>,
    mut sim_time: ResMut<SimTime>,
) {
    for (interaction, mut bg) in &mut q {
        bg.0 = match *interaction {
            Interaction::Pressed => {
                sim_time.sec += FutureBtn::SECS;
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

fn sky_plugin(app: &mut App) {
    app.get_sub_app_mut(RenderApp)
        .expect("No RenderApp")
        .add_systems(RenderStartup, setup_sky)
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

fn setup_sky(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    rd: Res<RenderDevice>,
) {
    commands.init_resource::<SpecializedRenderPipelines<SkyPipelineSpecializer>>();
    commands.insert_resource(SkyPipelineSpecializer {
        frag: asset_server.load("shaders/sky.wgsl"),
        vert: fullscreen_shader.to_vertex_state(),
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
                    (0, uniform_buffer::<SimTime>(false)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                ),
            ),
        ),
    });
}

#[derive(Resource)]
struct SkyPipelineSpecializer {
    frag: Handle<Shader>,
    vert: VertexState,
    layout: BindGroupLayout,
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
    sim_time: Res<SimTime>,
    rd: Res<RenderDevice>,
    rq: Res<RenderQueue>,
    specializer: Res<SkyPipelineSpecializer>,
    view_uniforms: Res<ViewUniforms>,
    mut commands: Commands,
) {
    let view_bindings = view_uniforms
        .uniforms
        .binding()
        .expect("Could not create view bindings for sky bind group");
    let mut sim_time_buffer = UniformBuffer::from(sim_time.clone());
    sim_time_buffer.write_buffer(&rd, &rq);
    let sim_time_binding = sim_time_buffer
        .binding()
        .expect("Could not create SimTime binding");
    let bind_group = rd.create_bind_group(
        "sky_bind_group",
        &specializer.layout,
        &BindGroupEntries::with_indices(((0, sim_time_binding), (3, view_bindings))),
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

fn water_plugin(app: &mut App) {
    app.get_sub_app_mut(RenderApp)
        .expect("No RenderApp")
        .add_systems(RenderStartup, setup_water)
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

fn setup_water(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    rd: Res<RenderDevice>,
) {
    commands.init_resource::<SpecializedRenderPipelines<WaterPipelineSpecializer>>();
    commands.insert_resource(WaterPipelineSpecializer {
        frag: asset_server.load("shaders/water.wgsl"),
        vert: fullscreen_shader.to_vertex_state(),
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
                    (4, uniform_buffer::<SimTime>(false)),
                ),
            ),
        ),
        sampler: rd.create_sampler(&default()),
    });
}

#[derive(Resource)]
struct WaterPipelineSpecializer {
    frag: Handle<Shader>,
    vert: VertexState,
    layout: BindGroupLayout,
    sampler: Sampler,
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
        let mut sim_time_buffer = UniformBuffer::from(world.resource::<SimTime>().clone());
        sim_time_buffer.write_buffer(
            render_context.render_device(),
            world.resource::<RenderQueue>(),
        );
        let sim_time_binding = sim_time_buffer
            .binding()
            .expect("Could not create SimTime binding");
        let bind_group = render_context.render_device().create_bind_group(
            "water_bind_group",
            &pipeline_specializer.layout,
            &BindGroupEntries::with_indices((
                (0, view_depth_texture.view()),
                (1, post_process.source),
                (2, &pipeline_specializer.sampler),
                (3, view_bindings),
                (4, sim_time_binding),
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
