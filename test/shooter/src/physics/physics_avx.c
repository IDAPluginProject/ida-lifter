/*
 * Advanced Physics System with AVX Batch Processing - Implementation
 * Stress-tests AVX lifter with extensive SIMD operations throughout.
 */

#include "physics/physics_avx.h"
#include "level/level.h"
#include "combat/bullet.h"
#include "math/avx_math.h"
#include "math/sse_math.h"
#include <stdlib.h>
#include <string.h>

/* Define max speed if not defined elsewhere */
#ifndef ENTITY_MAX_SPEED
#define ENTITY_MAX_SPEED PLAYER_RUN_SPEED
#endif

/* Static storage for physics state */
static PhysicsBatch g_physics_batch ALIGN32;
static SpatialHashGrid g_spatial_grid ALIGN32;
static CollisionResults g_collision_results ALIGN32;

/* ==========================================================================
 * AVX ENTITY MOVEMENT PROCESSING
 * ========================================================================== */

/*
 * AVX batch stuck entity detection and resolution
 */
static void avx_detect_stuck_entities_8(
    const float* pos_x, const float* pos_y,
    const float* prev_x, const float* prev_y,
    const float* vel_x, const float* vel_y,
    int* stuck_counters,
    float threshold
) {
    __m256 px = _mm256_load_ps(pos_x);
    __m256 py = _mm256_load_ps(pos_y);
    __m256 ox = _mm256_load_ps(prev_x);
    __m256 oy = _mm256_load_ps(prev_y);
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);

    /* Calculate movement distance */
    __m256 dx = _mm256_sub_ps(px, ox);
    __m256 dy = _mm256_sub_ps(py, oy);
    __m256 move_dist_sq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));

    /* Calculate desired movement */
    __m256 vel_sq = _mm256_fmadd_ps(vx, vx, _mm256_mul_ps(vy, vy));

    /* Check if stuck (moving less than expected) */
    __m256 thresh = _mm256_set1_ps(threshold * threshold);
    __m256 has_velocity = _mm256_cmp_ps(vel_sq, _mm256_set1_ps(0.001f), _CMP_GT_OQ);
    __m256 not_moving = _mm256_cmp_ps(move_dist_sq, thresh, _CMP_LT_OQ);
    __m256 is_stuck = _mm256_and_ps(has_velocity, not_moving);

    int stuck_mask = _mm256_movemask_ps(is_stuck);
    for (int i = 0; i < 8; i++) {
        if (stuck_mask & (1 << i)) {
            stuck_counters[i]++;
        } else {
            stuck_counters[i] = 0;
        }
    }
}

/*
 * AVX batch wall collision resolution
 */
static void avx_resolve_wall_collision_8(
    float* pos_x, float* pos_y,
    float* vel_x, float* vel_y,
    const Level* level,
    const float* radius
) {
    __m256 px = _mm256_load_ps(pos_x);
    __m256 py = _mm256_load_ps(pos_y);
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);
    __m256 rad = _mm256_load_ps(radius);

    /* Calculate tile coordinates for each entity */
    __m256i tile_x = _mm256_cvtps_epi32(px);
    __m256i tile_y = _mm256_cvtps_epi32(py);

    /* Check bounds */
    __m256i zero = _mm256_setzero_si256();
    __m256i max_x = _mm256_set1_epi32(LEVEL_WIDTH - 1);
    __m256i max_y = _mm256_set1_epi32(LEVEL_HEIGHT - 1);

    tile_x = _mm256_max_epi32(tile_x, zero);
    tile_x = _mm256_min_epi32(tile_x, max_x);
    tile_y = _mm256_max_epi32(tile_y, zero);
    tile_y = _mm256_min_epi32(tile_y, max_y);

    /* Calculate tile index */
    __m256i tile_idx = _mm256_add_epi32(
        tile_x,
        _mm256_mullo_epi32(tile_y, _mm256_set1_epi32(LEVEL_WIDTH))
    );

    /* Extract for scalar lookup */
    ALIGN32 int indices[8];
    ALIGN32 int tx[8], ty[8];
    _mm256_store_si256((__m256i*)indices, tile_idx);
    _mm256_store_si256((__m256i*)tx, tile_x);
    _mm256_store_si256((__m256i*)ty, tile_y);

    /* Check walls and compute adjustments */
    ALIGN32 float adj_x[8] = {0}, adj_y[8] = {0};
    ALIGN32 float vel_mul_x[8] = {1,1,1,1,1,1,1,1};
    ALIGN32 float vel_mul_y[8] = {1,1,1,1,1,1,1,1};
    ALIGN32 float r[8];
    _mm256_store_ps(r, rad);
    ALIGN32 float p_x[8], p_y[8];
    _mm256_store_ps(p_x, px);
    _mm256_store_ps(p_y, py);

    for (int i = 0; i < 8; i++) {
        int cx = tx[i];
        int cy = ty[i];

        /* Check adjacent tiles for collision */
        /* Left */
        if (cx > 0 && level->tiles[(cx-1) + cy * LEVEL_WIDTH] == TILE_WALL) {
            float wall_edge = (float)cx;
            if (p_x[i] - r[i] < wall_edge) {
                adj_x[i] = wall_edge + r[i] - p_x[i] + 0.01f;
                vel_mul_x[i] = -0.3f;
            }
        }
        /* Right */
        if (cx < LEVEL_WIDTH-1 && level->tiles[(cx+1) + cy * LEVEL_WIDTH] == TILE_WALL) {
            float wall_edge = (float)(cx + 1);
            if (p_x[i] + r[i] > wall_edge) {
                adj_x[i] = wall_edge - r[i] - p_x[i] - 0.01f;
                vel_mul_x[i] = -0.3f;
            }
        }
        /* Down */
        if (cy > 0 && level->tiles[cx + (cy-1) * LEVEL_WIDTH] == TILE_WALL) {
            float wall_edge = (float)cy;
            if (p_y[i] - r[i] < wall_edge) {
                adj_y[i] = wall_edge + r[i] - p_y[i] + 0.01f;
                vel_mul_y[i] = -0.3f;
            }
        }
        /* Up */
        if (cy < LEVEL_HEIGHT-1 && level->tiles[cx + (cy+1) * LEVEL_WIDTH] == TILE_WALL) {
            float wall_edge = (float)(cy + 1);
            if (p_y[i] + r[i] > wall_edge) {
                adj_y[i] = wall_edge - r[i] - p_y[i] - 0.01f;
                vel_mul_y[i] = -0.3f;
            }
        }
    }

    /* Apply adjustments using AVX */
    __m256 ax = _mm256_load_ps(adj_x);
    __m256 ay = _mm256_load_ps(adj_y);
    __m256 vmx = _mm256_load_ps(vel_mul_x);
    __m256 vmy = _mm256_load_ps(vel_mul_y);

    px = _mm256_add_ps(px, ax);
    py = _mm256_add_ps(py, ay);
    vx = _mm256_mul_ps(vx, vmx);
    vy = _mm256_mul_ps(vy, vmy);

    _mm256_store_ps(pos_x, px);
    _mm256_store_ps(pos_y, py);
    _mm256_store_ps(vel_x, vx);
    _mm256_store_ps(vel_y, vy);
}

/*
 * AVX batch steering force application
 */
static void avx_apply_steering_8(
    float* vel_x, float* vel_y,
    const float* steer_x, const float* steer_y,
    const float* max_force,
    const float* max_speed
) {
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);
    __m256 sx = _mm256_load_ps(steer_x);
    __m256 sy = _mm256_load_ps(steer_y);
    __m256 mf = _mm256_load_ps(max_force);
    __m256 ms = _mm256_load_ps(max_speed);

    /* Clamp steering force magnitude */
    __m256 steer_mag_sq = _mm256_fmadd_ps(sx, sx, _mm256_mul_ps(sy, sy));
    __m256 steer_mag = _mm256_sqrt_ps(_mm256_max_ps(steer_mag_sq, _mm256_set1_ps(0.0001f)));
    __m256 scale = _mm256_div_ps(mf, _mm256_max_ps(steer_mag, mf));
    scale = _mm256_min_ps(scale, _mm256_set1_ps(1.0f));

    sx = _mm256_mul_ps(sx, scale);
    sy = _mm256_mul_ps(sy, scale);

    /* Apply steering */
    vx = _mm256_add_ps(vx, sx);
    vy = _mm256_add_ps(vy, sy);

    /* Clamp velocity magnitude */
    __m256 vel_mag_sq = _mm256_fmadd_ps(vx, vx, _mm256_mul_ps(vy, vy));
    __m256 vel_mag = _mm256_sqrt_ps(_mm256_max_ps(vel_mag_sq, _mm256_set1_ps(0.0001f)));
    __m256 vel_scale = _mm256_div_ps(ms, _mm256_max_ps(vel_mag, ms));
    vel_scale = _mm256_min_ps(vel_scale, _mm256_set1_ps(1.0f));

    vx = _mm256_mul_ps(vx, vel_scale);
    vy = _mm256_mul_ps(vy, vel_scale);

    _mm256_store_ps(vel_x, vx);
    _mm256_store_ps(vel_y, vy);
}

/* ==========================================================================
 * AVX BULLET PROCESSING
 * ========================================================================== */

/*
 * AVX batch bullet movement with wall collision
 */
static int avx_update_bullets_batch(
    GameState* game,
    int start_idx,
    int count
) {
    if (count <= 0) return 0;

    /* Pad to 8 */
    ALIGN32 float bx[8], by[8], bvx[8], bvy[8];
    ALIGN32 int active[8];

    for (int i = 0; i < count && i < 8; i++) {
        Bullet* b = &game->bullets[start_idx + i];
        bx[i] = b->x;
        by[i] = b->y;
        bvx[i] = b->vx;
        bvy[i] = b->vy;
        active[i] = b->active ? -1 : 0;
    }
    for (int i = count; i < 8; i++) {
        bx[i] = by[i] = bvx[i] = bvy[i] = 0;
        active[i] = 0;
    }

    /* Load into AVX */
    __m256 px = _mm256_load_ps(bx);
    __m256 py = _mm256_load_ps(by);
    __m256 vx = _mm256_load_ps(bvx);
    __m256 vy = _mm256_load_ps(bvy);

    /* Calculate new positions */
    __m256 new_x = _mm256_add_ps(px, vx);
    __m256 new_y = _mm256_add_ps(py, vy);

    /* Check bounds */
    __m256 zero = _mm256_setzero_ps();
    __m256 max_x = _mm256_set1_ps((float)(LEVEL_WIDTH - 1));
    __m256 max_y = _mm256_set1_ps((float)(LEVEL_HEIGHT - 1));

    __m256 in_bounds = _mm256_and_ps(
        _mm256_and_ps(
            _mm256_cmp_ps(new_x, zero, _CMP_GE_OQ),
            _mm256_cmp_ps(new_x, max_x, _CMP_LT_OQ)
        ),
        _mm256_and_ps(
            _mm256_cmp_ps(new_y, zero, _CMP_GE_OQ),
            _mm256_cmp_ps(new_y, max_y, _CMP_LT_OQ)
        )
    );

    /* Convert to tile coordinates for wall check */
    __m256i tile_x = _mm256_cvtps_epi32(new_x);
    __m256i tile_y = _mm256_cvtps_epi32(new_y);
    __m256i tile_idx = _mm256_add_epi32(
        tile_x,
        _mm256_mullo_epi32(tile_y, _mm256_set1_epi32(LEVEL_WIDTH))
    );

    /* Extract for scalar wall check */
    ALIGN32 int indices[8];
    _mm256_store_si256((__m256i*)indices, tile_idx);

    ALIGN32 float new_x_arr[8], new_y_arr[8];
    _mm256_store_ps(new_x_arr, new_x);
    _mm256_store_ps(new_y_arr, new_y);

    int bounds_mask = _mm256_movemask_ps(in_bounds);

    int deactivated = 0;
    for (int i = 0; i < count && i < 8; i++) {
        Bullet* b = &game->bullets[start_idx + i];
        if (!b->active) continue;

        bool hit_wall = false;
        if (!(bounds_mask & (1 << i))) {
            hit_wall = true;
        } else if (indices[i] >= 0 && indices[i] < LEVEL_SIZE) {
            if (game->level.tiles[indices[i]] == TILE_WALL) {
                hit_wall = true;
            }
        }

        if (hit_wall) {
            b->active = false;
            deactivated++;
        } else {
            b->x = new_x_arr[i];
            b->y = new_y_arr[i];
        }
    }

    return deactivated;
}

/*
 * AVX batch bullet-entity collision detection
 */
static void avx_bullet_entity_collision_8(
    GameState* game,
    int bullet_start,
    int bullet_count
) {
    if (bullet_count <= 0) return;

    ALIGN32 float bx[8], by[8];
    ALIGN32 float bdmg[8];
    ALIGN32 int bteam[8], bowner[8], bactive[8];

    /* Load bullet batch */
    for (int i = 0; i < bullet_count && i < 8; i++) {
        Bullet* b = &game->bullets[bullet_start + i];
        bx[i] = b->x;
        by[i] = b->y;
        bdmg[i] = b->damage;
        bteam[i] = b->team;
        bowner[i] = b->owner_id;
        bactive[i] = b->active ? 1 : 0;
    }
    for (int i = bullet_count; i < 8; i++) {
        bx[i] = by[i] = -1000.0f;
        bdmg[i] = 0;
        bteam[i] = -1;
        bowner[i] = -1;
        bactive[i] = 0;
    }

    __m256 bullet_x = _mm256_load_ps(bx);
    __m256 bullet_y = _mm256_load_ps(by);

    /* Test against each entity */
    for (int ei = 0; ei < game->entity_count; ei++) {
        Entity* e = &game->entities[ei];
        if (!e->alive) continue;

        /* Broadcast entity position */
        __m256 ex = _mm256_set1_ps(e->x);
        __m256 ey = _mm256_set1_ps(e->y);

        /* Calculate distance squared */
        __m256 dx = _mm256_sub_ps(bullet_x, ex);
        __m256 dy = _mm256_sub_ps(bullet_y, ey);
        __m256 dist_sq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));

        /* Check hit (radius = 1.0) */
        __m256 hit = _mm256_cmp_ps(dist_sq, _mm256_set1_ps(1.0f), _CMP_LT_OQ);
        int hit_mask = _mm256_movemask_ps(hit);

        /* Process hits */
        for (int i = 0; i < bullet_count && i < 8; i++) {
            if (!(hit_mask & (1 << i))) continue;
            if (!bactive[i]) continue;
            if (bteam[i] == e->team) continue;
            if (bowner[i] == ei) continue;

            /* Apply damage */
            e->health -= bdmg[i];

            /* Record damage source */
            if (bowner[i] >= 0 && bowner[i] < game->entity_count) {
                Entity* shooter = &game->entities[bowner[i]];
                e->last_damage_x = shooter->x;
                e->last_damage_y = shooter->y;
            } else {
                e->last_damage_x = e->x - game->bullets[bullet_start + i].vx * 10.0f;
                e->last_damage_y = e->y - game->bullets[bullet_start + i].vy * 10.0f;
            }
            e->damage_react_timer = 60;

            /* Deactivate bullet */
            game->bullets[bullet_start + i].active = false;
            bactive[i] = 0;

            /* Check death */
            if (e->health <= 0) {
                e->alive = false;
                e->state = STATE_DEAD;
            }
            break;
        }
    }
}

/* ==========================================================================
 * AVX ENTITY-ENTITY COLLISION
 * ========================================================================== */

/*
 * AVX batch entity separation (soft collision)
 */
static void avx_separate_entities_8(
    float* pos_x, float* pos_y,
    float* vel_x, float* vel_y,
    int entity_a,
    const float* other_x, const float* other_y,
    float separation_radius,
    float separation_force
) {
    __m256 ax = _mm256_set1_ps(pos_x[entity_a]);
    __m256 ay = _mm256_set1_ps(pos_y[entity_a]);
    __m256 ox = _mm256_load_ps(other_x);
    __m256 oy = _mm256_load_ps(other_y);

    /* Direction from others to self */
    __m256 dx = _mm256_sub_ps(ax, ox);
    __m256 dy = _mm256_sub_ps(ay, oy);

    /* Distance */
    __m256 dist_sq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
    __m256 dist = _mm256_sqrt_ps(_mm256_max_ps(dist_sq, _mm256_set1_ps(0.0001f)));

    /* Check if within separation radius */
    __m256 rad = _mm256_set1_ps(separation_radius);
    __m256 in_range = _mm256_cmp_ps(dist, rad, _CMP_LT_OQ);
    __m256 not_self = _mm256_cmp_ps(dist, _mm256_set1_ps(0.01f), _CMP_GT_OQ);
    __m256 should_separate = _mm256_and_ps(in_range, not_self);

    /* Calculate separation strength (stronger when closer) */
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 strength = _mm256_sub_ps(one, _mm256_div_ps(dist, rad));
    strength = _mm256_mul_ps(strength, _mm256_set1_ps(separation_force));
    strength = _mm256_and_ps(strength, should_separate);

    /* Normalize direction */
    __m256 inv_dist = _mm256_div_ps(one, dist);
    __m256 nx = _mm256_mul_ps(dx, inv_dist);
    __m256 ny = _mm256_mul_ps(dy, inv_dist);

    /* Calculate force */
    __m256 fx = _mm256_mul_ps(nx, strength);
    __m256 fy = _mm256_mul_ps(ny, strength);

    /* Sum forces using horizontal add */
    __m256 sum_x = fx;
    __m256 sum_y = fy;

    /* Horizontal sum (AVX2) */
    sum_x = _mm256_hadd_ps(sum_x, sum_x);
    sum_x = _mm256_hadd_ps(sum_x, sum_x);
    sum_y = _mm256_hadd_ps(sum_y, sum_y);
    sum_y = _mm256_hadd_ps(sum_y, sum_y);

    /* Extract final sum */
    float total_x = _mm_cvtss_f32(_mm256_castps256_ps128(sum_x)) +
                    _mm_cvtss_f32(_mm256_extractf128_ps(sum_x, 1));
    float total_y = _mm_cvtss_f32(_mm256_castps256_ps128(sum_y)) +
                    _mm_cvtss_f32(_mm256_extractf128_ps(sum_y, 1));

    /* Apply to entity */
    vel_x[entity_a] += total_x;
    vel_y[entity_a] += total_y;
}

/* ==========================================================================
 * AVX SOUND PROPAGATION
 * ========================================================================== */

/*
 * AVX batch sound attenuation calculation
 */
static void avx_calculate_sound_attenuation_8(
    float source_x, float source_y,
    const float* listener_x, const float* listener_y,
    float base_volume,
    float* volumes_out
) {
    __m256 sx = _mm256_set1_ps(source_x);
    __m256 sy = _mm256_set1_ps(source_y);
    __m256 lx = _mm256_load_ps(listener_x);
    __m256 ly = _mm256_load_ps(listener_y);

    /* Distance */
    __m256 dx = _mm256_sub_ps(lx, sx);
    __m256 dy = _mm256_sub_ps(ly, sy);
    __m256 dist_sq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
    __m256 dist = _mm256_sqrt_ps(_mm256_max_ps(dist_sq, _mm256_set1_ps(1.0f)));

    /* Inverse square falloff with distance */
    __m256 base = _mm256_set1_ps(base_volume);
    __m256 attenuation = _mm256_div_ps(base, _mm256_mul_ps(dist, dist));

    /* Clamp to [0, 1] */
    attenuation = _mm256_min_ps(attenuation, _mm256_set1_ps(1.0f));
    attenuation = _mm256_max_ps(attenuation, _mm256_setzero_ps());

    _mm256_store_ps(volumes_out, attenuation);
}

/* ==========================================================================
 * AVX FOOTSTEP SOUND GENERATION
 * ========================================================================== */

/*
 * AVX batch footstep calculation
 */
static void avx_calculate_footsteps_8(
    const float* vel_x, const float* vel_y,
    const int* is_running, const int* is_crouching,
    float* sound_radius_out
) {
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);

    /* Movement speed */
    __m256 speed_sq = _mm256_fmadd_ps(vx, vx, _mm256_mul_ps(vy, vy));
    __m256 speed = _mm256_sqrt_ps(speed_sq);

    /* Threshold for footsteps */
    __m256 moving = _mm256_cmp_ps(speed, _mm256_set1_ps(0.05f), _CMP_GT_OQ);

    /* Base sound levels */
    __m256 walk_base = _mm256_set1_ps(SOUND_WALK_BASE);
    __m256 run_base = _mm256_set1_ps(SOUND_RUN_BASE);

    /* Load running/crouching flags */
    ALIGN32 float run_f[8], crouch_f[8];
    for (int i = 0; i < 8; i++) {
        run_f[i] = is_running[i] ? 1.0f : 0.0f;
        crouch_f[i] = is_crouching[i] ? 0.3f : 1.0f;
    }
    __m256 running = _mm256_load_ps(run_f);
    __m256 crouch_mult = _mm256_load_ps(crouch_f);

    /* Select base sound */
    __m256 run_mask = _mm256_cmp_ps(running, _mm256_set1_ps(0.5f), _CMP_GT_OQ);
    __m256 base_sound = _mm256_blendv_ps(walk_base, run_base, run_mask);

    /* Apply crouch modifier */
    base_sound = _mm256_mul_ps(base_sound, crouch_mult);

    /* Scale by speed */
    __m256 radius = _mm256_mul_ps(base_sound, _mm256_add_ps(_mm256_set1_ps(1.0f), speed));

    /* Zero out if not moving */
    radius = _mm256_and_ps(radius, moving);

    _mm256_store_ps(sound_radius_out, radius);
}

/* ==========================================================================
 * MAIN PHYSICS UPDATE FUNCTION
 * ========================================================================== */

void update_physics_avx(GameState* game) {
    /* Initialize physics batch from game state */
    phys_avx_init_batch(&g_physics_batch, game);

    /* === ENTITY PHYSICS === */

    /* Process entities in batches of 8 */
    for (int i = 0; i + 8 <= game->entity_count; i += 8) {
        /* Check for stuck entities */
        ALIGN32 float prev_x[8], prev_y[8];
        ALIGN32 int stuck[8];
        for (int j = 0; j < 8; j++) {
            prev_x[j] = game->entities[i + j].prev_x;
            prev_y[j] = game->entities[i + j].prev_y;
            stuck[j] = game->entities[i + j].stuck_counter;
        }

        avx_detect_stuck_entities_8(
            &g_physics_batch.pos_x[i], &g_physics_batch.pos_y[i],
            prev_x, prev_y,
            &g_physics_batch.vel_x[i], &g_physics_batch.vel_y[i],
            stuck, 0.05f
        );

        /* Store stuck counters back */
        for (int j = 0; j < 8; j++) {
            game->entities[i + j].stuck_counter = stuck[j];
        }

        /* Apply friction */
        phys_avx_apply_friction_8(
            &g_physics_batch.vel_x[i],
            &g_physics_batch.vel_y[i],
            &g_physics_batch.friction[i]
        );

        /* Integrate position */
        phys_avx_integrate_position_8(
            &g_physics_batch.pos_x[i], &g_physics_batch.pos_y[i],
            &g_physics_batch.vel_x[i], &g_physics_batch.vel_y[i],
            1.0f  /* dt */
        );

        /* Wall collision */
        avx_resolve_wall_collision_8(
            &g_physics_batch.pos_x[i], &g_physics_batch.pos_y[i],
            &g_physics_batch.vel_x[i], &g_physics_batch.vel_y[i],
            &game->level,
            &g_physics_batch.radius[i]
        );

        /* Clamp velocity */
        phys_avx_clamp_velocity_8(
            &g_physics_batch.vel_x[i],
            &g_physics_batch.vel_y[i],
            ENTITY_MAX_SPEED
        );
    }

    /* Handle remaining entities (scalar) */
    for (int i = (game->entity_count / 8) * 8; i < game->entity_count; i++) {
        Entity* e = &game->entities[i];
        if (!e->alive) continue;

        /* Apply friction */
        e->vx *= 0.85f;
        e->vy *= 0.85f;

        /* Update position */
        float new_x = e->x + e->vx;
        float new_y = e->y + e->vy;

        /* Wall collision */
        if (is_walkable(&game->level, (int)new_x, (int)new_y)) {
            e->x = new_x;
            e->y = new_y;
        } else {
            if (is_walkable(&game->level, (int)new_x, (int)e->y)) {
                e->x = new_x;
            }
            if (is_walkable(&game->level, (int)e->x, (int)new_y)) {
                e->y = new_y;
            }
        }
    }

    /* Update spatial grid for entity-entity collision */
    phys_avx_update_spatial_grid(&g_spatial_grid, &g_physics_batch);

    /* Broad phase collision detection */
    phys_avx_broad_phase(&g_spatial_grid, &g_physics_batch, &g_collision_results);

    /* Narrow phase and resolve collisions */
    for (int i = 0; i + 8 <= g_collision_results.count; i += 8) {
        phys_avx_narrow_phase_8(&g_physics_batch, &g_collision_results.pairs[i], 8);
    }

    /* Apply entity separation (soft collision) */
    for (int i = 0; i < game->entity_count; i++) {
        if (!game->entities[i].alive) continue;

        ALIGN32 float other_x[8], other_y[8];
        int other_count = 0;

        /* Gather nearby entities */
        for (int j = 0; j < game->entity_count && other_count < 8; j++) {
            if (i == j || !game->entities[j].alive) continue;

            float dx = g_physics_batch.pos_x[j] - g_physics_batch.pos_x[i];
            float dy = g_physics_batch.pos_y[j] - g_physics_batch.pos_y[i];
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < 4.0f) {  /* Within separation range */
                other_x[other_count] = g_physics_batch.pos_x[j];
                other_y[other_count] = g_physics_batch.pos_y[j];
                other_count++;
            }
        }

        if (other_count > 0) {
            /* Pad remaining slots */
            for (int k = other_count; k < 8; k++) {
                other_x[k] = g_physics_batch.pos_x[i] + 1000.0f;  /* Far away */
                other_y[k] = g_physics_batch.pos_y[i] + 1000.0f;
            }

            avx_separate_entities_8(
                g_physics_batch.pos_x, g_physics_batch.pos_y,
                g_physics_batch.vel_x, g_physics_batch.vel_y,
                i,
                other_x, other_y,
                1.5f,  /* separation radius */
                0.1f   /* separation force */
            );
        }
    }

    /* Write back to game state */
    for (int i = 0; i < game->entity_count; i++) {
        Entity* e = &game->entities[i];
        e->prev_x = e->x;
        e->prev_y = e->y;
        e->x = g_physics_batch.pos_x[i];
        e->y = g_physics_batch.pos_y[i];
        e->vx = g_physics_batch.vel_x[i];
        e->vy = g_physics_batch.vel_y[i];
    }

    /* === BULLET PHYSICS === */

    /* Update bullets in batches of 8 */
    for (int i = 0; i < game->bullet_count; i += 8) {
        int count = (game->bullet_count - i < 8) ? game->bullet_count - i : 8;
        avx_update_bullets_batch(game, i, count);
    }

    /* Bullet-entity collision in batches */
    for (int i = 0; i < game->bullet_count; i += 8) {
        int count = (game->bullet_count - i < 8) ? game->bullet_count - i : 8;
        avx_bullet_entity_collision_8(game, i, count);
    }

    /* Compact bullet array */
    int write_idx = 0;
    for (int i = 0; i < game->bullet_count; i++) {
        if (game->bullets[i].active) {
            if (i != write_idx) {
                game->bullets[write_idx] = game->bullets[i];
            }
            write_idx++;
        }
    }
    game->bullet_count = write_idx;

    /* === FOOTSTEP SOUNDS === */

    /* Calculate footstep sounds in batches */
    for (int i = 0; i + 8 <= game->entity_count; i += 8) {
        ALIGN32 int running[8], crouching[8];
        ALIGN32 float sound_radius[8];

        for (int j = 0; j < 8; j++) {
            running[j] = game->entities[i + j].is_running ? 1 : 0;
            crouching[j] = game->entities[i + j].is_crouching ? 1 : 0;
        }

        avx_calculate_footsteps_8(
            &g_physics_batch.vel_x[i], &g_physics_batch.vel_y[i],
            running, crouching,
            sound_radius
        );

        /* Propagate sounds */
        for (int j = 0; j < 8; j++) {
            Entity* e = &game->entities[i + j];
            if (!e->alive || sound_radius[j] < 0.1f) continue;

            e->steps_since_sound++;
            float speed = sqrtf(e->vx * e->vx + e->vy * e->vy);
            if (e->steps_since_sound > (int)(5.0f / (speed + 0.1f))) {
                propagate_sound(game, e->x, e->y, sound_radius[j]);
                e->steps_since_sound = 0;
            }
        }
    }
}

/* ==========================================================================
 * AVX BULLET UPDATE WITH CCD
 * ========================================================================== */

void update_bullets_avx(GameState* game) {
    CCDResult8 ccd_result ALIGN32;

    for (int i = 0; i < game->bullet_count; i += 8) {
        int count = (game->bullet_count - i < 8) ? game->bullet_count - i : 8;

        ALIGN32 float start_x[8], start_y[8], end_x[8], end_y[8];

        /* Load bullet start/end positions */
        for (int j = 0; j < count; j++) {
            Bullet* b = &game->bullets[i + j];
            start_x[j] = b->x;
            start_y[j] = b->y;
            end_x[j] = b->x + b->vx;
            end_y[j] = b->y + b->vy;
        }
        for (int j = count; j < 8; j++) {
            start_x[j] = start_y[j] = end_x[j] = end_y[j] = 0;
        }

        /* Run CCD sweep */
        phys_avx_ccd_sweep_8(start_x, start_y, end_x, end_y, &game->level, &ccd_result);

        /* Apply results */
        for (int j = 0; j < count; j++) {
            Bullet* b = &game->bullets[i + j];
            if (!b->active) continue;

            if (ccd_result.hit_mask & (1 << j)) {
                /* Hit wall - stop at hit position */
                b->x = ccd_result.hit_x[j];
                b->y = ccd_result.hit_y[j];
                b->active = false;
            } else {
                /* No wall hit - move to end position */
                b->x = end_x[j];
                b->y = end_y[j];
            }
        }
    }
}

/* ==========================================================================
 * AVX COLLISION RESOLUTION
 * ========================================================================== */

void resolve_collisions_avx(GameState* game, PhysicsBatch* batch, CollisionResults* results) {
    if (!results || results->count == 0) return;

    /* Process collision pairs in batches */
    for (int i = 0; i + 8 <= results->count; i += 8) {
        ALIGN32 float vel_x1[8], vel_y1[8], vel_x2[8], vel_y2[8];
        ALIGN32 float mass1[8], mass2[8];
        ALIGN32 float nx[8], ny[8], bounce[8];

        /* Load collision pair data */
        for (int j = 0; j < 8; j++) {
            CollisionPair* p = &results->pairs[i + j];

            /* Skip if no actual collision */
            if (p->penetration_x == 0.0f && p->penetration_y == 0.0f) {
                vel_x1[j] = vel_y1[j] = vel_x2[j] = vel_y2[j] = 0;
                mass1[j] = mass2[j] = 1.0f;
                nx[j] = 1.0f;
                ny[j] = 0.0f;
                bounce[j] = 0.0f;
                continue;
            }

            int a = p->entity_a;
            int b = p->entity_b;

            vel_x1[j] = batch->vel_x[a];
            vel_y1[j] = batch->vel_y[a];
            vel_x2[j] = batch->vel_x[b];
            vel_y2[j] = batch->vel_y[b];
            mass1[j] = batch->mass[a];
            mass2[j] = batch->mass[b];
            nx[j] = p->normal_x;
            ny[j] = p->normal_y;
            bounce[j] = (batch->bounce[a] + batch->bounce[b]) * 0.5f;
        }

        /* Apply elastic response */
        phys_avx_elastic_response_8(
            vel_x1, vel_y1, mass1,
            vel_x2, vel_y2, mass2,
            nx, ny, bounce
        );

        /* Write back velocities */
        for (int j = 0; j < 8; j++) {
            CollisionPair* p = &results->pairs[i + j];
            if (p->penetration_x == 0.0f && p->penetration_y == 0.0f) continue;

            batch->vel_x[p->entity_a] = vel_x1[j];
            batch->vel_y[p->entity_a] = vel_y1[j];
            batch->vel_x[p->entity_b] = vel_x2[j];
            batch->vel_y[p->entity_b] = vel_y2[j];
        }

        /* Apply position separation */
        ALIGN32 float pos_x1[8], pos_y1[8], pos_x2[8], pos_y2[8];
        ALIGN32 float pen[8];

        for (int j = 0; j < 8; j++) {
            CollisionPair* p = &results->pairs[i + j];
            int a = p->entity_a;
            int b = p->entity_b;

            pos_x1[j] = batch->pos_x[a];
            pos_y1[j] = batch->pos_y[a];
            pos_x2[j] = batch->pos_x[b];
            pos_y2[j] = batch->pos_y[b];
            pen[j] = sqrtf(p->penetration_x * p->penetration_x +
                          p->penetration_y * p->penetration_y);
        }

        phys_avx_separate_8(
            pos_x1, pos_y1,
            pos_x2, pos_y2,
            nx, ny, pen,
            mass1, mass2
        );

        /* Write back positions */
        for (int j = 0; j < 8; j++) {
            CollisionPair* p = &results->pairs[i + j];
            if (p->penetration_x == 0.0f && p->penetration_y == 0.0f) continue;

            batch->pos_x[p->entity_a] = pos_x1[j];
            batch->pos_y[p->entity_a] = pos_y1[j];
            batch->pos_x[p->entity_b] = pos_x2[j];
            batch->pos_y[p->entity_b] = pos_y2[j];
        }
    }
}
