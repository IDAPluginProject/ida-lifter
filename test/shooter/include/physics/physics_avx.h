/*
 * Advanced Physics System with AVX Batch Processing
 * Processes entities in batches of 8 for maximum SIMD utilization.
 * Designed to stress-test AVX lifter with extensive vector operations.
 */

#ifndef SHOOTER_PHYSICS_AVX_H
#define SHOOTER_PHYSICS_AVX_H

#include "../types.h"
#include "../config.h"
#include <immintrin.h>

/* ==========================================================================
 * SPATIAL HASH GRID FOR BROAD-PHASE COLLISION
 * ========================================================================== */

#define SPATIAL_CELL_SIZE 4.0f
#define SPATIAL_GRID_WIDTH  (LEVEL_WIDTH / 4)
#define SPATIAL_GRID_HEIGHT (LEVEL_HEIGHT / 4)
#define SPATIAL_GRID_SIZE   (SPATIAL_GRID_WIDTH * SPATIAL_GRID_HEIGHT)
#define MAX_ENTITIES_PER_CELL 16

typedef struct {
    int entity_ids[MAX_ENTITIES_PER_CELL];
    int count;
} SpatialCell;

typedef struct ALIGN32 {
    SpatialCell cells[SPATIAL_GRID_SIZE];
    /* Cached entity AABBs for AVX processing */
    float min_x[MAX_ENTITIES];
    float min_y[MAX_ENTITIES];
    float max_x[MAX_ENTITIES];
    float max_y[MAX_ENTITIES];
} SpatialHashGrid;

/* ==========================================================================
 * COLLISION RESULTS
 * ========================================================================== */

typedef struct {
    int entity_a;
    int entity_b;
    float penetration_x;
    float penetration_y;
    float normal_x;
    float normal_y;
} CollisionPair;

typedef struct ALIGN32 {
    CollisionPair pairs[MAX_ENTITIES * 4];
    int count;
} CollisionResults;

/* ==========================================================================
 * PHYSICS STATE FOR BATCH PROCESSING
 * ========================================================================== */

typedef struct ALIGN32 {
    /* Position batch (8-wide) */
    float pos_x[MAX_ENTITIES];
    float pos_y[MAX_ENTITIES];

    /* Velocity batch */
    float vel_x[MAX_ENTITIES];
    float vel_y[MAX_ENTITIES];

    /* Acceleration batch */
    float acc_x[MAX_ENTITIES];
    float acc_y[MAX_ENTITIES];

    /* Entity properties */
    float mass[MAX_ENTITIES];
    float radius[MAX_ENTITIES];
    float friction[MAX_ENTITIES];
    float bounce[MAX_ENTITIES];

    /* Collision flags */
    uint32_t collision_mask[MAX_ENTITIES];
    uint32_t active_flags[(MAX_ENTITIES + 31) / 32];

    int entity_count;
} PhysicsBatch;

/* ==========================================================================
 * CONTINUOUS COLLISION DETECTION (CCD)
 * ========================================================================== */

typedef struct ALIGN32 {
    float toi[8];           /* Time of impact (0-1) */
    float hit_x[8];         /* Hit position X */
    float hit_y[8];         /* Hit position Y */
    float normal_x[8];      /* Hit normal X */
    float normal_y[8];      /* Hit normal Y */
    int hit_entity[8];      /* Entity that was hit (-1 if wall) */
    uint32_t hit_mask;      /* Bitmask of which rays hit */
} CCDResult8;

/* ==========================================================================
 * AVX BATCH PHYSICS OPERATIONS
 * ========================================================================== */

/*
 * Initialize physics batch from game entities
 */
static inline void phys_avx_init_batch(PhysicsBatch* batch, const GameState* game) {
    batch->entity_count = game->entity_count;

    /* Process in batches of 8 */
    int i = 0;
    for (; i + 8 <= game->entity_count; i += 8) {
        /* Load positions */
        __m256 px = _mm256_set_ps(
            game->entities[i+7].x, game->entities[i+6].x,
            game->entities[i+5].x, game->entities[i+4].x,
            game->entities[i+3].x, game->entities[i+2].x,
            game->entities[i+1].x, game->entities[i+0].x
        );
        __m256 py = _mm256_set_ps(
            game->entities[i+7].y, game->entities[i+6].y,
            game->entities[i+5].y, game->entities[i+4].y,
            game->entities[i+3].y, game->entities[i+2].y,
            game->entities[i+1].y, game->entities[i+0].y
        );

        /* Load velocities */
        __m256 vx = _mm256_set_ps(
            game->entities[i+7].vx, game->entities[i+6].vx,
            game->entities[i+5].vx, game->entities[i+4].vx,
            game->entities[i+3].vx, game->entities[i+2].vx,
            game->entities[i+1].vx, game->entities[i+0].vx
        );
        __m256 vy = _mm256_set_ps(
            game->entities[i+7].vy, game->entities[i+6].vy,
            game->entities[i+5].vy, game->entities[i+4].vy,
            game->entities[i+3].vy, game->entities[i+2].vy,
            game->entities[i+1].vy, game->entities[i+0].vy
        );

        _mm256_store_ps(&batch->pos_x[i], px);
        _mm256_store_ps(&batch->pos_y[i], py);
        _mm256_store_ps(&batch->vel_x[i], vx);
        _mm256_store_ps(&batch->vel_y[i], vy);

        /* Initialize with default physics properties */
        __m256 default_mass = _mm256_set1_ps(1.0f);
        __m256 default_radius = _mm256_set1_ps(0.5f);
        __m256 default_friction = _mm256_set1_ps(0.85f);
        __m256 default_bounce = _mm256_set1_ps(0.3f);

        _mm256_store_ps(&batch->mass[i], default_mass);
        _mm256_store_ps(&batch->radius[i], default_radius);
        _mm256_store_ps(&batch->friction[i], default_friction);
        _mm256_store_ps(&batch->bounce[i], default_bounce);

        /* Zero acceleration */
        __m256 zero = _mm256_setzero_ps();
        _mm256_store_ps(&batch->acc_x[i], zero);
        _mm256_store_ps(&batch->acc_y[i], zero);
    }

    /* Handle remaining entities */
    for (; i < game->entity_count; i++) {
        batch->pos_x[i] = game->entities[i].x;
        batch->pos_y[i] = game->entities[i].y;
        batch->vel_x[i] = game->entities[i].vx;
        batch->vel_y[i] = game->entities[i].vy;
        batch->acc_x[i] = 0.0f;
        batch->acc_y[i] = 0.0f;
        batch->mass[i] = 1.0f;
        batch->radius[i] = 0.5f;
        batch->friction[i] = 0.85f;
        batch->bounce[i] = 0.3f;
    }
}

/*
 * AVX batch integration - update velocities from accelerations
 */
static inline void phys_avx_integrate_velocity_8(
    float* vel_x, float* vel_y,
    const float* acc_x, const float* acc_y,
    float dt
) {
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);
    __m256 ax = _mm256_load_ps(acc_x);
    __m256 ay = _mm256_load_ps(acc_y);
    __m256 dt_vec = _mm256_set1_ps(dt);

    /* v += a * dt */
    vx = _mm256_fmadd_ps(ax, dt_vec, vx);
    vy = _mm256_fmadd_ps(ay, dt_vec, vy);

    _mm256_store_ps(vel_x, vx);
    _mm256_store_ps(vel_y, vy);
}

/*
 * AVX batch integration - update positions from velocities
 */
static inline void phys_avx_integrate_position_8(
    float* pos_x, float* pos_y,
    const float* vel_x, const float* vel_y,
    float dt
) {
    __m256 px = _mm256_load_ps(pos_x);
    __m256 py = _mm256_load_ps(pos_y);
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);
    __m256 dt_vec = _mm256_set1_ps(dt);

    /* p += v * dt */
    px = _mm256_fmadd_ps(vx, dt_vec, px);
    py = _mm256_fmadd_ps(vy, dt_vec, py);

    _mm256_store_ps(pos_x, px);
    _mm256_store_ps(pos_y, py);
}

/*
 * AVX batch friction application
 */
static inline void phys_avx_apply_friction_8(
    float* vel_x, float* vel_y,
    const float* friction
) {
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);
    __m256 fric = _mm256_load_ps(friction);

    vx = _mm256_mul_ps(vx, fric);
    vy = _mm256_mul_ps(vy, fric);

    _mm256_store_ps(vel_x, vx);
    _mm256_store_ps(vel_y, vy);
}

/*
 * AVX batch velocity clamping
 */
static inline void phys_avx_clamp_velocity_8(
    float* vel_x, float* vel_y,
    float max_speed
) {
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);
    __m256 max_sq = _mm256_set1_ps(max_speed * max_speed);

    /* Calculate speed squared */
    __m256 speed_sq = _mm256_fmadd_ps(vx, vx, _mm256_mul_ps(vy, vy));

    /* Calculate scale factor: min(1.0, max_speed / speed) */
    __m256 speed = _mm256_sqrt_ps(speed_sq);
    __m256 max_spd = _mm256_set1_ps(max_speed);
    __m256 scale = _mm256_div_ps(max_spd, _mm256_max_ps(speed, _mm256_set1_ps(0.0001f)));
    scale = _mm256_min_ps(scale, _mm256_set1_ps(1.0f));

    /* Apply scale only where speed exceeds max */
    __m256 needs_clamp = _mm256_cmp_ps(speed_sq, max_sq, _CMP_GT_OQ);
    __m256 one = _mm256_set1_ps(1.0f);
    scale = _mm256_blendv_ps(one, scale, needs_clamp);

    vx = _mm256_mul_ps(vx, scale);
    vy = _mm256_mul_ps(vy, scale);

    _mm256_store_ps(vel_x, vx);
    _mm256_store_ps(vel_y, vy);
}

/*
 * AVX batch distance squared calculation (8 pairs)
 */
static inline __m256 phys_avx_distance_sq_8(
    __m256 x1, __m256 y1,
    __m256 x2, __m256 y2
) {
    __m256 dx = _mm256_sub_ps(x2, x1);
    __m256 dy = _mm256_sub_ps(y2, y1);
    return _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
}

/*
 * AVX batch circle-circle collision test
 * Returns bitmask of colliding pairs
 */
static inline uint32_t phys_avx_circle_collision_8(
    __m256 x1, __m256 y1, __m256 r1,
    __m256 x2, __m256 y2, __m256 r2
) {
    __m256 dist_sq = phys_avx_distance_sq_8(x1, y1, x2, y2);
    __m256 rad_sum = _mm256_add_ps(r1, r2);
    __m256 rad_sum_sq = _mm256_mul_ps(rad_sum, rad_sum);

    __m256 collision = _mm256_cmp_ps(dist_sq, rad_sum_sq, _CMP_LT_OQ);
    return (uint32_t)_mm256_movemask_ps(collision);
}

/*
 * AVX batch AABB-AABB collision test
 */
static inline uint32_t phys_avx_aabb_collision_8(
    __m256 min_x1, __m256 min_y1, __m256 max_x1, __m256 max_y1,
    __m256 min_x2, __m256 min_y2, __m256 max_x2, __m256 max_y2
) {
    /* Test: !(max1 < min2 || min1 > max2) for both axes */
    __m256 no_overlap_x1 = _mm256_cmp_ps(max_x1, min_x2, _CMP_LT_OQ);
    __m256 no_overlap_x2 = _mm256_cmp_ps(min_x1, max_x2, _CMP_GT_OQ);
    __m256 no_overlap_y1 = _mm256_cmp_ps(max_y1, min_y2, _CMP_LT_OQ);
    __m256 no_overlap_y2 = _mm256_cmp_ps(min_y1, max_y2, _CMP_GT_OQ);

    __m256 no_overlap_x = _mm256_or_ps(no_overlap_x1, no_overlap_x2);
    __m256 no_overlap_y = _mm256_or_ps(no_overlap_y1, no_overlap_y2);
    __m256 no_overlap = _mm256_or_ps(no_overlap_x, no_overlap_y);

    /* Invert: collision if NOT no_overlap */
    __m256 collision = _mm256_xor_ps(no_overlap, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    return (uint32_t)_mm256_movemask_ps(collision);
}

/*
 * AVX batch point-in-AABB test
 */
static inline uint32_t phys_avx_point_in_aabb_8(
    __m256 px, __m256 py,
    __m256 min_x, __m256 min_y,
    __m256 max_x, __m256 max_y
) {
    __m256 in_x = _mm256_and_ps(
        _mm256_cmp_ps(px, min_x, _CMP_GE_OQ),
        _mm256_cmp_ps(px, max_x, _CMP_LE_OQ)
    );
    __m256 in_y = _mm256_and_ps(
        _mm256_cmp_ps(py, min_y, _CMP_GE_OQ),
        _mm256_cmp_ps(py, max_y, _CMP_LE_OQ)
    );
    __m256 inside = _mm256_and_ps(in_x, in_y);
    return (uint32_t)_mm256_movemask_ps(inside);
}

/*
 * AVX batch ray-AABB intersection (for CCD)
 * Returns time of intersection (0-1) or >1 if no hit
 */
static inline void phys_avx_ray_aabb_8(
    __m256 ray_ox, __m256 ray_oy,      /* Ray origin */
    __m256 ray_dx, __m256 ray_dy,      /* Ray direction */
    __m256 box_min_x, __m256 box_min_y,
    __m256 box_max_x, __m256 box_max_y,
    __m256* t_out,                      /* Time of hit */
    __m256* hit_mask                    /* Which rays hit */
) {
    __m256 epsilon = _mm256_set1_ps(0.0001f);
    __m256 inf = _mm256_set1_ps(1e30f);
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);

    /* Compute inverse direction (avoiding divide by zero) */
    __m256 inv_dx = _mm256_div_ps(one, _mm256_blendv_ps(ray_dx, epsilon,
        _mm256_cmp_ps(_mm256_and_ps(ray_dx, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))), epsilon, _CMP_LT_OQ)));
    __m256 inv_dy = _mm256_div_ps(one, _mm256_blendv_ps(ray_dy, epsilon,
        _mm256_cmp_ps(_mm256_and_ps(ray_dy, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))), epsilon, _CMP_LT_OQ)));

    /* Calculate t values for each slab */
    __m256 t1x = _mm256_mul_ps(_mm256_sub_ps(box_min_x, ray_ox), inv_dx);
    __m256 t2x = _mm256_mul_ps(_mm256_sub_ps(box_max_x, ray_ox), inv_dx);
    __m256 t1y = _mm256_mul_ps(_mm256_sub_ps(box_min_y, ray_oy), inv_dy);
    __m256 t2y = _mm256_mul_ps(_mm256_sub_ps(box_max_y, ray_oy), inv_dy);

    /* Get min/max for each axis */
    __m256 tmin_x = _mm256_min_ps(t1x, t2x);
    __m256 tmax_x = _mm256_max_ps(t1x, t2x);
    __m256 tmin_y = _mm256_min_ps(t1y, t2y);
    __m256 tmax_y = _mm256_max_ps(t1y, t2y);

    /* Get overall tmin/tmax */
    __m256 tmin = _mm256_max_ps(tmin_x, tmin_y);
    __m256 tmax = _mm256_min_ps(tmax_x, tmax_y);

    /* Check for valid intersection */
    __m256 valid = _mm256_and_ps(
        _mm256_cmp_ps(tmax, tmin, _CMP_GE_OQ),
        _mm256_cmp_ps(tmax, zero, _CMP_GE_OQ)
    );
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(tmin, one, _CMP_LE_OQ));

    /* Clamp tmin to [0, 1] */
    tmin = _mm256_max_ps(tmin, zero);
    tmin = _mm256_min_ps(tmin, one);

    /* Output: inf if no hit, tmin if hit */
    *t_out = _mm256_blendv_ps(inf, tmin, valid);
    *hit_mask = valid;
}

/*
 * AVX batch collision response (elastic)
 */
static inline void phys_avx_elastic_response_8(
    float* vel_x1, float* vel_y1, const float* mass1,
    float* vel_x2, float* vel_y2, const float* mass2,
    const float* nx, const float* ny,  /* Collision normal */
    const float* bounce
) {
    __m256 vx1 = _mm256_load_ps(vel_x1);
    __m256 vy1 = _mm256_load_ps(vel_y1);
    __m256 vx2 = _mm256_load_ps(vel_x2);
    __m256 vy2 = _mm256_load_ps(vel_y2);
    __m256 m1 = _mm256_load_ps(mass1);
    __m256 m2 = _mm256_load_ps(mass2);
    __m256 normal_x = _mm256_load_ps(nx);
    __m256 normal_y = _mm256_load_ps(ny);
    __m256 restitution = _mm256_load_ps(bounce);

    /* Relative velocity */
    __m256 rel_vx = _mm256_sub_ps(vx1, vx2);
    __m256 rel_vy = _mm256_sub_ps(vy1, vy2);

    /* Relative velocity along normal */
    __m256 rel_vel_n = _mm256_fmadd_ps(rel_vx, normal_x, _mm256_mul_ps(rel_vy, normal_y));

    /* Only resolve if objects are approaching */
    __m256 approaching = _mm256_cmp_ps(rel_vel_n, _mm256_setzero_ps(), _CMP_LT_OQ);

    /* Calculate impulse scalar */
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 e_plus_1 = _mm256_add_ps(one, restitution);
    __m256 mass_sum = _mm256_add_ps(m1, m2);
    __m256 inv_mass_sum = _mm256_div_ps(one, mass_sum);

    __m256 j = _mm256_mul_ps(_mm256_mul_ps(e_plus_1, rel_vel_n), inv_mass_sum);
    j = _mm256_blendv_ps(_mm256_setzero_ps(), j, approaching);

    /* Apply impulse */
    __m256 j_nx = _mm256_mul_ps(j, normal_x);
    __m256 j_ny = _mm256_mul_ps(j, normal_y);

    __m256 inv_m1 = _mm256_div_ps(one, m1);
    __m256 inv_m2 = _mm256_div_ps(one, m2);

    vx1 = _mm256_fnmadd_ps(j_nx, inv_m1, vx1);
    vy1 = _mm256_fnmadd_ps(j_ny, inv_m1, vy1);
    vx2 = _mm256_fmadd_ps(j_nx, inv_m2, vx2);
    vy2 = _mm256_fmadd_ps(j_ny, inv_m2, vy2);

    _mm256_store_ps(vel_x1, vx1);
    _mm256_store_ps(vel_y1, vy1);
    _mm256_store_ps(vel_x2, vx2);
    _mm256_store_ps(vel_y2, vy2);
}

/*
 * AVX batch position correction (separation)
 */
static inline void phys_avx_separate_8(
    float* pos_x1, float* pos_y1,
    float* pos_x2, float* pos_y2,
    const float* nx, const float* ny,
    const float* penetration,
    const float* mass1, const float* mass2
) {
    __m256 px1 = _mm256_load_ps(pos_x1);
    __m256 py1 = _mm256_load_ps(pos_y1);
    __m256 px2 = _mm256_load_ps(pos_x2);
    __m256 py2 = _mm256_load_ps(pos_y2);
    __m256 normal_x = _mm256_load_ps(nx);
    __m256 normal_y = _mm256_load_ps(ny);
    __m256 pen = _mm256_load_ps(penetration);
    __m256 m1 = _mm256_load_ps(mass1);
    __m256 m2 = _mm256_load_ps(mass2);

    /* Calculate separation ratio based on mass */
    __m256 mass_sum = _mm256_add_ps(m1, m2);
    __m256 ratio1 = _mm256_div_ps(m2, mass_sum);
    __m256 ratio2 = _mm256_div_ps(m1, mass_sum);

    /* Apply separation */
    __m256 sep1 = _mm256_mul_ps(pen, ratio1);
    __m256 sep2 = _mm256_mul_ps(pen, ratio2);

    px1 = _mm256_fmadd_ps(sep1, normal_x, px1);
    py1 = _mm256_fmadd_ps(sep1, normal_y, py1);
    px2 = _mm256_fnmadd_ps(sep2, normal_x, px2);
    py2 = _mm256_fnmadd_ps(sep2, normal_y, py2);

    _mm256_store_ps(pos_x1, px1);
    _mm256_store_ps(pos_y1, py1);
    _mm256_store_ps(pos_x2, px2);
    _mm256_store_ps(pos_y2, py2);
}

/*
 * AVX batch wall collision (axis-aligned)
 */
static inline void phys_avx_wall_collision_8(
    float* pos_x, float* pos_y,
    float* vel_x, float* vel_y,
    float min_x, float min_y,
    float max_x, float max_y,
    float bounce_factor
) {
    __m256 px = _mm256_load_ps(pos_x);
    __m256 py = _mm256_load_ps(pos_y);
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);

    __m256 bmin_x = _mm256_set1_ps(min_x);
    __m256 bmin_y = _mm256_set1_ps(min_y);
    __m256 bmax_x = _mm256_set1_ps(max_x);
    __m256 bmax_y = _mm256_set1_ps(max_y);
    __m256 bounce = _mm256_set1_ps(-bounce_factor);

    /* Left wall */
    __m256 hit_left = _mm256_cmp_ps(px, bmin_x, _CMP_LT_OQ);
    px = _mm256_blendv_ps(px, bmin_x, hit_left);
    vx = _mm256_blendv_ps(vx, _mm256_mul_ps(vx, bounce), hit_left);

    /* Right wall */
    __m256 hit_right = _mm256_cmp_ps(px, bmax_x, _CMP_GT_OQ);
    px = _mm256_blendv_ps(px, bmax_x, hit_right);
    vx = _mm256_blendv_ps(vx, _mm256_mul_ps(vx, bounce), hit_right);

    /* Bottom wall */
    __m256 hit_bottom = _mm256_cmp_ps(py, bmin_y, _CMP_LT_OQ);
    py = _mm256_blendv_ps(py, bmin_y, hit_bottom);
    vy = _mm256_blendv_ps(vy, _mm256_mul_ps(vy, bounce), hit_bottom);

    /* Top wall */
    __m256 hit_top = _mm256_cmp_ps(py, bmax_y, _CMP_GT_OQ);
    py = _mm256_blendv_ps(py, bmax_y, hit_top);
    vy = _mm256_blendv_ps(vy, _mm256_mul_ps(vy, bounce), hit_top);

    _mm256_store_ps(pos_x, px);
    _mm256_store_ps(pos_y, py);
    _mm256_store_ps(vel_x, vx);
    _mm256_store_ps(vel_y, vy);
}

/*
 * AVX batch gravity application
 */
static inline void phys_avx_apply_gravity_8(
    float* acc_y,
    float gravity
) {
    __m256 ay = _mm256_load_ps(acc_y);
    __m256 g = _mm256_set1_ps(gravity);
    ay = _mm256_add_ps(ay, g);
    _mm256_store_ps(acc_y, ay);
}

/*
 * AVX batch force application (F = ma -> a = F/m)
 */
static inline void phys_avx_apply_force_8(
    float* acc_x, float* acc_y,
    const float* force_x, const float* force_y,
    const float* mass
) {
    __m256 ax = _mm256_load_ps(acc_x);
    __m256 ay = _mm256_load_ps(acc_y);
    __m256 fx = _mm256_load_ps(force_x);
    __m256 fy = _mm256_load_ps(force_y);
    __m256 m = _mm256_load_ps(mass);

    /* a += F / m */
    __m256 inv_m = _mm256_div_ps(_mm256_set1_ps(1.0f), m);
    ax = _mm256_fmadd_ps(fx, inv_m, ax);
    ay = _mm256_fmadd_ps(fy, inv_m, ay);

    _mm256_store_ps(acc_x, ax);
    _mm256_store_ps(acc_y, ay);
}

/*
 * AVX batch kinetic energy calculation
 */
static inline __m256 phys_avx_kinetic_energy_8(
    const float* vel_x, const float* vel_y,
    const float* mass
) {
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);
    __m256 m = _mm256_load_ps(mass);

    /* KE = 0.5 * m * v^2 */
    __m256 v_sq = _mm256_fmadd_ps(vx, vx, _mm256_mul_ps(vy, vy));
    __m256 half = _mm256_set1_ps(0.5f);
    return _mm256_mul_ps(_mm256_mul_ps(half, m), v_sq);
}

/*
 * AVX batch momentum calculation
 */
static inline void phys_avx_momentum_8(
    const float* vel_x, const float* vel_y,
    const float* mass,
    float* mom_x, float* mom_y
) {
    __m256 vx = _mm256_load_ps(vel_x);
    __m256 vy = _mm256_load_ps(vel_y);
    __m256 m = _mm256_load_ps(mass);

    /* p = m * v */
    __m256 px = _mm256_mul_ps(m, vx);
    __m256 py = _mm256_mul_ps(m, vy);

    _mm256_store_ps(mom_x, px);
    _mm256_store_ps(mom_y, py);
}

/* ==========================================================================
 * SPATIAL HASHING WITH AVX
 * ========================================================================== */

/*
 * AVX batch spatial hash computation
 */
static inline __m256i phys_avx_spatial_hash_8(
    __m256 pos_x, __m256 pos_y
) {
    __m256 inv_cell = _mm256_set1_ps(1.0f / SPATIAL_CELL_SIZE);
    __m256i grid_w = _mm256_set1_epi32(SPATIAL_GRID_WIDTH);

    /* Convert position to cell coordinates */
    __m256i cell_x = _mm256_cvtps_epi32(_mm256_mul_ps(pos_x, inv_cell));
    __m256i cell_y = _mm256_cvtps_epi32(_mm256_mul_ps(pos_y, inv_cell));

    /* Clamp to grid bounds */
    __m256i zero = _mm256_setzero_si256();
    __m256i max_x = _mm256_set1_epi32(SPATIAL_GRID_WIDTH - 1);
    __m256i max_y = _mm256_set1_epi32(SPATIAL_GRID_HEIGHT - 1);

    cell_x = _mm256_max_epi32(cell_x, zero);
    cell_x = _mm256_min_epi32(cell_x, max_x);
    cell_y = _mm256_max_epi32(cell_y, zero);
    cell_y = _mm256_min_epi32(cell_y, max_y);

    /* Calculate hash: y * width + x */
    return _mm256_add_epi32(_mm256_mullo_epi32(cell_y, grid_w), cell_x);
}

/*
 * Update spatial hash grid for all entities
 */
static inline void phys_avx_update_spatial_grid(
    SpatialHashGrid* grid,
    const PhysicsBatch* batch
) {
    /* Clear grid */
    for (int i = 0; i < SPATIAL_GRID_SIZE; i++) {
        grid->cells[i].count = 0;
    }

    /* Update AABBs and insert into grid */
    ALIGN32 int cell_indices[8];

    for (int i = 0; i + 8 <= batch->entity_count; i += 8) {
        __m256 px = _mm256_load_ps(&batch->pos_x[i]);
        __m256 py = _mm256_load_ps(&batch->pos_y[i]);
        __m256 rad = _mm256_load_ps(&batch->radius[i]);

        /* Compute AABBs */
        __m256 min_x = _mm256_sub_ps(px, rad);
        __m256 min_y = _mm256_sub_ps(py, rad);
        __m256 max_x = _mm256_add_ps(px, rad);
        __m256 max_y = _mm256_add_ps(py, rad);

        _mm256_store_ps(&grid->min_x[i], min_x);
        _mm256_store_ps(&grid->min_y[i], min_y);
        _mm256_store_ps(&grid->max_x[i], max_x);
        _mm256_store_ps(&grid->max_y[i], max_y);

        /* Compute cell indices */
        __m256i cells = phys_avx_spatial_hash_8(px, py);
        _mm256_store_si256((__m256i*)cell_indices, cells);

        /* Insert into cells */
        for (int j = 0; j < 8; j++) {
            int cell_idx = cell_indices[j];
            SpatialCell* cell = &grid->cells[cell_idx];
            if (cell->count < MAX_ENTITIES_PER_CELL) {
                cell->entity_ids[cell->count++] = i + j;
            }
        }
    }

    /* Handle remaining entities */
    for (int i = (batch->entity_count / 8) * 8; i < batch->entity_count; i++) {
        float px = batch->pos_x[i];
        float py = batch->pos_y[i];
        float rad = batch->radius[i];

        grid->min_x[i] = px - rad;
        grid->min_y[i] = py - rad;
        grid->max_x[i] = px + rad;
        grid->max_y[i] = py + rad;

        int cell_x = (int)(px / SPATIAL_CELL_SIZE);
        int cell_y = (int)(py / SPATIAL_CELL_SIZE);
        cell_x = cell_x < 0 ? 0 : (cell_x >= SPATIAL_GRID_WIDTH ? SPATIAL_GRID_WIDTH - 1 : cell_x);
        cell_y = cell_y < 0 ? 0 : (cell_y >= SPATIAL_GRID_HEIGHT ? SPATIAL_GRID_HEIGHT - 1 : cell_y);

        int cell_idx = cell_y * SPATIAL_GRID_WIDTH + cell_x;
        SpatialCell* cell = &grid->cells[cell_idx];
        if (cell->count < MAX_ENTITIES_PER_CELL) {
            cell->entity_ids[cell->count++] = i;
        }
    }
}

/* ==========================================================================
 * BROAD PHASE COLLISION DETECTION
 * ========================================================================== */

/*
 * AVX broad phase - find potential collision pairs using spatial hash
 */
static inline int phys_avx_broad_phase(
    const SpatialHashGrid* grid,
    const PhysicsBatch* batch,
    CollisionResults* results
) {
    results->count = 0;

    ALIGN32 float test_min_x[8], test_min_y[8], test_max_x[8], test_max_y[8];

    for (int cell_idx = 0; cell_idx < SPATIAL_GRID_SIZE; cell_idx++) {
        const SpatialCell* cell = &grid->cells[cell_idx];
        if (cell->count < 2) continue;

        /* Test all pairs within cell */
        for (int i = 0; i < cell->count; i++) {
            int id_a = cell->entity_ids[i];

            /* Load entity A's AABB */
            __m256 a_min_x = _mm256_set1_ps(grid->min_x[id_a]);
            __m256 a_min_y = _mm256_set1_ps(grid->min_y[id_a]);
            __m256 a_max_x = _mm256_set1_ps(grid->max_x[id_a]);
            __m256 a_max_y = _mm256_set1_ps(grid->max_y[id_a]);

            /* Test against remaining entities in batches of 8 */
            for (int j = i + 1; j < cell->count; j += 8) {
                int count = (cell->count - j < 8) ? cell->count - j : 8;

                /* Load batch of entity B AABBs */
                for (int k = 0; k < count; k++) {
                    int id_b = cell->entity_ids[j + k];
                    test_min_x[k] = grid->min_x[id_b];
                    test_min_y[k] = grid->min_y[id_b];
                    test_max_x[k] = grid->max_x[id_b];
                    test_max_y[k] = grid->max_y[id_b];
                }
                for (int k = count; k < 8; k++) {
                    test_min_x[k] = 1e30f;
                    test_min_y[k] = 1e30f;
                    test_max_x[k] = -1e30f;
                    test_max_y[k] = -1e30f;
                }

                __m256 b_min_x = _mm256_load_ps(test_min_x);
                __m256 b_min_y = _mm256_load_ps(test_min_y);
                __m256 b_max_x = _mm256_load_ps(test_max_x);
                __m256 b_max_y = _mm256_load_ps(test_max_y);

                /* Test AABB overlap */
                uint32_t hits = phys_avx_aabb_collision_8(
                    a_min_x, a_min_y, a_max_x, a_max_y,
                    b_min_x, b_min_y, b_max_x, b_max_y
                );

                /* Record collision pairs */
                while (hits) {
                    int bit = __builtin_ctz(hits);
                    hits &= hits - 1;

                    if (bit < count && results->count < MAX_ENTITIES * 4) {
                        CollisionPair* pair = &results->pairs[results->count++];
                        pair->entity_a = id_a;
                        pair->entity_b = cell->entity_ids[j + bit];
                    }
                }
            }
        }
    }

    return results->count;
}

/* ==========================================================================
 * NARROW PHASE COLLISION DETECTION
 * ========================================================================== */

/*
 * AVX narrow phase - precise circle collision with penetration info
 */
static inline void phys_avx_narrow_phase_8(
    const PhysicsBatch* batch,
    CollisionPair* pairs,  /* 8 pairs to test */
    int pair_count
) {
    ALIGN32 float ax[8], ay[8], bx[8], by[8], ar[8], br[8];

    /* Load pair data */
    for (int i = 0; i < pair_count; i++) {
        int id_a = pairs[i].entity_a;
        int id_b = pairs[i].entity_b;
        ax[i] = batch->pos_x[id_a];
        ay[i] = batch->pos_y[id_a];
        bx[i] = batch->pos_x[id_b];
        by[i] = batch->pos_y[id_b];
        ar[i] = batch->radius[id_a];
        br[i] = batch->radius[id_b];
    }
    for (int i = pair_count; i < 8; i++) {
        ax[i] = ay[i] = bx[i] = by[i] = 0.0f;
        ar[i] = br[i] = 0.0f;
    }

    __m256 pax = _mm256_load_ps(ax);
    __m256 pay = _mm256_load_ps(ay);
    __m256 pbx = _mm256_load_ps(bx);
    __m256 pby = _mm256_load_ps(by);
    __m256 rad_a = _mm256_load_ps(ar);
    __m256 rad_b = _mm256_load_ps(br);

    /* Calculate collision normal and penetration */
    __m256 dx = _mm256_sub_ps(pbx, pax);
    __m256 dy = _mm256_sub_ps(pby, pay);
    __m256 dist_sq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
    __m256 dist = _mm256_sqrt_ps(_mm256_max_ps(dist_sq, _mm256_set1_ps(0.0001f)));

    /* Normalize direction */
    __m256 inv_dist = _mm256_div_ps(_mm256_set1_ps(1.0f), dist);
    __m256 nx = _mm256_mul_ps(dx, inv_dist);
    __m256 ny = _mm256_mul_ps(dy, inv_dist);

    /* Calculate penetration */
    __m256 rad_sum = _mm256_add_ps(rad_a, rad_b);
    __m256 penetration = _mm256_sub_ps(rad_sum, dist);

    /* Check for actual collision */
    __m256 collision = _mm256_cmp_ps(penetration, _mm256_setzero_ps(), _CMP_GT_OQ);

    /* Store results */
    ALIGN32 float out_nx[8], out_ny[8], out_pen[8];
    _mm256_store_ps(out_nx, nx);
    _mm256_store_ps(out_ny, ny);
    _mm256_store_ps(out_pen, penetration);

    int collision_mask = _mm256_movemask_ps(collision);
    for (int i = 0; i < pair_count; i++) {
        if (collision_mask & (1 << i)) {
            pairs[i].normal_x = out_nx[i];
            pairs[i].normal_y = out_ny[i];
            pairs[i].penetration_x = out_pen[i] * out_nx[i];
            pairs[i].penetration_y = out_pen[i] * out_ny[i];
        } else {
            /* Mark as no collision */
            pairs[i].penetration_x = 0.0f;
            pairs[i].penetration_y = 0.0f;
        }
    }
}

/* ==========================================================================
 * CONTINUOUS COLLISION DETECTION (CCD)
 * ========================================================================== */

/*
 * AVX batch CCD for bullets/fast objects against walls
 */
static inline void phys_avx_ccd_sweep_8(
    const float* start_x, const float* start_y,
    const float* end_x, const float* end_y,
    const Level* level,
    CCDResult8* result
) {
    __m256 sx = _mm256_load_ps(start_x);
    __m256 sy = _mm256_load_ps(start_y);
    __m256 ex = _mm256_load_ps(end_x);
    __m256 ey = _mm256_load_ps(end_y);

    /* Direction vector */
    __m256 dx = _mm256_sub_ps(ex, sx);
    __m256 dy = _mm256_sub_ps(ey, sy);

    /* Length of sweep */
    __m256 len_sq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
    __m256 len = _mm256_sqrt_ps(_mm256_max_ps(len_sq, _mm256_set1_ps(0.0001f)));

    /* Step size (approx 0.5 tile per step) */
    __m256 inv_len = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
    __m256 step = _mm256_mul_ps(_mm256_set1_ps(0.5f), inv_len);
    step = _mm256_min_ps(step, _mm256_set1_ps(0.1f));

    /* Initialize results */
    __m256 best_t = _mm256_set1_ps(1.0f);
    __m256 hit_x = ex;
    __m256 hit_y = ey;
    __m256 norm_x = _mm256_setzero_ps();
    __m256 norm_y = _mm256_setzero_ps();
    __m256 hit_found = _mm256_setzero_ps();

    /* March along ray */
    for (float t = 0.0f; t <= 1.0f; t += 0.05f) {
        __m256 t_vec = _mm256_set1_ps(t);
        __m256 px = _mm256_fmadd_ps(dx, t_vec, sx);
        __m256 py = _mm256_fmadd_ps(dy, t_vec, sy);

        /* Convert to tile coordinates */
        __m256i tile_x = _mm256_cvtps_epi32(px);
        __m256i tile_y = _mm256_cvtps_epi32(py);

        /* Check bounds */
        __m256i zero = _mm256_setzero_si256();
        __m256i max_x = _mm256_set1_epi32(LEVEL_WIDTH - 1);
        __m256i max_y = _mm256_set1_epi32(LEVEL_HEIGHT - 1);

        __m256i in_bounds_x = _mm256_and_si256(
            _mm256_cmpgt_epi32(tile_x, _mm256_sub_epi32(zero, _mm256_set1_epi32(1))),
            _mm256_cmpgt_epi32(max_x, _mm256_sub_epi32(tile_x, _mm256_set1_epi32(1)))
        );
        __m256i in_bounds_y = _mm256_and_si256(
            _mm256_cmpgt_epi32(tile_y, _mm256_sub_epi32(zero, _mm256_set1_epi32(1))),
            _mm256_cmpgt_epi32(max_y, _mm256_sub_epi32(tile_y, _mm256_set1_epi32(1)))
        );
        __m256i in_bounds = _mm256_and_si256(in_bounds_x, in_bounds_y);

        /* Calculate tile index */
        __m256i tile_idx = _mm256_add_epi32(
            tile_x,
            _mm256_mullo_epi32(tile_y, _mm256_set1_epi32(LEVEL_WIDTH))
        );

        /* Check tile values (scalar due to gather limitations) */
        ALIGN32 int indices[8];
        _mm256_store_si256((__m256i*)indices, tile_idx);

        ALIGN32 int bounds[8];
        _mm256_store_si256((__m256i*)bounds, in_bounds);

        ALIGN32 int wall_hits[8];
        for (int i = 0; i < 8; i++) {
            if (bounds[i] && indices[i] >= 0 && indices[i] < LEVEL_SIZE) {
                wall_hits[i] = (level->tiles[indices[i]] == TILE_WALL) ? -1 : 0;
            } else {
                wall_hits[i] = -1;  /* Out of bounds = wall */
            }
        }

        __m256 is_wall = _mm256_castsi256_ps(_mm256_load_si256((__m256i*)wall_hits));

        /* Update best hit (only if not already found a closer hit) */
        __m256 new_hit = _mm256_andnot_ps(hit_found, is_wall);
        __m256 update = _mm256_and_ps(new_hit, _mm256_cmp_ps(t_vec, best_t, _CMP_LT_OQ));

        best_t = _mm256_blendv_ps(best_t, t_vec, update);
        hit_x = _mm256_blendv_ps(hit_x, px, update);
        hit_y = _mm256_blendv_ps(hit_y, py, update);

        /* Simple normal approximation based on direction */
        __m256 abs_dx = _mm256_and_ps(dx, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
        __m256 abs_dy = _mm256_and_ps(dy, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
        __m256 x_dominant = _mm256_cmp_ps(abs_dx, abs_dy, _CMP_GT_OQ);

        __m256 sign_dx = _mm256_and_ps(dx, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
        __m256 sign_dy = _mm256_and_ps(dy, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));

        __m256 nx_new = _mm256_blendv_ps(_mm256_setzero_ps(),
            _mm256_xor_ps(_mm256_set1_ps(1.0f), sign_dx), x_dominant);
        __m256 ny_new = _mm256_blendv_ps(
            _mm256_xor_ps(_mm256_set1_ps(1.0f), sign_dy), _mm256_setzero_ps(), x_dominant);

        norm_x = _mm256_blendv_ps(norm_x, nx_new, update);
        norm_y = _mm256_blendv_ps(norm_y, ny_new, update);

        hit_found = _mm256_or_ps(hit_found, new_hit);
    }

    /* Store results */
    _mm256_store_ps(result->toi, best_t);
    _mm256_store_ps(result->hit_x, hit_x);
    _mm256_store_ps(result->hit_y, hit_y);
    _mm256_store_ps(result->normal_x, norm_x);
    _mm256_store_ps(result->normal_y, norm_y);
    result->hit_mask = (uint32_t)_mm256_movemask_ps(hit_found);

    for (int i = 0; i < 8; i++) {
        result->hit_entity[i] = -1;  /* Wall hit */
    }
}

/* ==========================================================================
 * VERLET INTEGRATION (FOR CONSTRAINTS)
 * ========================================================================== */

typedef struct ALIGN32 {
    float pos_x[MAX_ENTITIES];
    float pos_y[MAX_ENTITIES];
    float prev_x[MAX_ENTITIES];
    float prev_y[MAX_ENTITIES];
    float acc_x[MAX_ENTITIES];
    float acc_y[MAX_ENTITIES];
} VerletBatch;

/*
 * AVX batch Verlet integration step
 */
static inline void phys_avx_verlet_integrate_8(
    float* pos_x, float* pos_y,
    float* prev_x, float* prev_y,
    const float* acc_x, const float* acc_y,
    float dt_sq
) {
    __m256 px = _mm256_load_ps(pos_x);
    __m256 py = _mm256_load_ps(pos_y);
    __m256 ox = _mm256_load_ps(prev_x);
    __m256 oy = _mm256_load_ps(prev_y);
    __m256 ax = _mm256_load_ps(acc_x);
    __m256 ay = _mm256_load_ps(acc_y);
    __m256 dt2 = _mm256_set1_ps(dt_sq);

    /* new_pos = 2*pos - prev_pos + acc*dt^2 */
    __m256 two = _mm256_set1_ps(2.0f);

    __m256 new_x = _mm256_fmadd_ps(two, px, _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), ox, _mm256_mul_ps(ax, dt2)));
    __m256 new_y = _mm256_fmadd_ps(two, py, _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), oy, _mm256_mul_ps(ay, dt2)));

    /* Update prev to current, current to new */
    _mm256_store_ps(prev_x, px);
    _mm256_store_ps(prev_y, py);
    _mm256_store_ps(pos_x, new_x);
    _mm256_store_ps(pos_y, new_y);
}

/*
 * AVX batch distance constraint (for rope/chain physics)
 */
static inline void phys_avx_distance_constraint_8(
    float* pos_x1, float* pos_y1,
    float* pos_x2, float* pos_y2,
    float rest_length,
    float stiffness
) {
    __m256 x1 = _mm256_load_ps(pos_x1);
    __m256 y1 = _mm256_load_ps(pos_y1);
    __m256 x2 = _mm256_load_ps(pos_x2);
    __m256 y2 = _mm256_load_ps(pos_y2);

    __m256 dx = _mm256_sub_ps(x2, x1);
    __m256 dy = _mm256_sub_ps(y2, y1);

    __m256 dist_sq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
    __m256 dist = _mm256_sqrt_ps(_mm256_max_ps(dist_sq, _mm256_set1_ps(0.0001f)));

    __m256 rest = _mm256_set1_ps(rest_length);
    __m256 stiff = _mm256_set1_ps(stiffness * 0.5f);

    /* Calculate correction */
    __m256 diff = _mm256_sub_ps(dist, rest);
    __m256 inv_dist = _mm256_div_ps(_mm256_set1_ps(1.0f), dist);

    __m256 corr_x = _mm256_mul_ps(_mm256_mul_ps(diff, stiff), _mm256_mul_ps(dx, inv_dist));
    __m256 corr_y = _mm256_mul_ps(_mm256_mul_ps(diff, stiff), _mm256_mul_ps(dy, inv_dist));

    /* Apply correction (equal mass assumption) */
    x1 = _mm256_add_ps(x1, corr_x);
    y1 = _mm256_add_ps(y1, corr_y);
    x2 = _mm256_sub_ps(x2, corr_x);
    y2 = _mm256_sub_ps(y2, corr_y);

    _mm256_store_ps(pos_x1, x1);
    _mm256_store_ps(pos_y1, y1);
    _mm256_store_ps(pos_x2, x2);
    _mm256_store_ps(pos_y2, y2);
}

/* ==========================================================================
 * MAIN PHYSICS UPDATE
 * ========================================================================== */

/*
 * Full physics update with AVX batch processing
 */
void update_physics_avx(GameState* game);

/*
 * Update bullet physics with CCD
 */
void update_bullets_avx(GameState* game);

/*
 * Resolve all collisions
 */
void resolve_collisions_avx(GameState* game, PhysicsBatch* batch, CollisionResults* results);

#endif /* SHOOTER_PHYSICS_AVX_H */
