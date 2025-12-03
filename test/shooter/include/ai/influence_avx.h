/*
 * Advanced Influence Map System with AVX Batch Updates
 * Spatial reasoning system for tactical AI decisions.
 * Uses AVX/AVX2 for batch map updates and queries.
 */

#ifndef SHOOTER_INFLUENCE_AVX_H
#define SHOOTER_INFLUENCE_AVX_H

#include <immintrin.h>
#include <string.h>
#include "../config.h"
#include "../types.h"
#include "../math/avx_math.h"

/* ==========================================================================
 * INFLUENCE MAP CONFIGURATION
 * ========================================================================== */

#define INF_CELL_SIZE 1.0f        /* World units per cell */
#define INF_MAP_WIDTH LEVEL_WIDTH
#define INF_MAP_HEIGHT LEVEL_HEIGHT
#define INF_MAP_SIZE (INF_MAP_WIDTH * INF_MAP_HEIGHT)

/* Update frequencies */
#define INF_THREAT_UPDATE_FREQ 5      /* Frames between threat updates */
#define INF_COVER_UPDATE_FREQ 60      /* Frames between cover recalc */
#define INF_VISIBILITY_UPDATE_FREQ 15 /* Frames between visibility updates */

/* ==========================================================================
 * INFLUENCE MAP LAYERS
 * ========================================================================== */

typedef struct __attribute__((aligned(32))) {
    /* Threat layer - danger from enemies */
    float threat[INF_MAP_SIZE];

    /* Visibility layer - how exposed each cell is */
    float visibility[INF_MAP_SIZE];

    /* Cover layer - quality of cover at each cell */
    float cover[INF_MAP_SIZE];

    /* Sound layer - recent sound events */
    float sound[INF_MAP_SIZE];

    /* Path cost layer - tactical movement cost */
    float path_cost[INF_MAP_SIZE];

    /* Combined tactical value */
    float tactical[INF_MAP_SIZE];

    /* Update tracking */
    int threat_last_update;
    int cover_last_update;
    int visibility_last_update;
} InfluenceMapAdvanced;

/* ==========================================================================
 * AVX BATCH MAP OPERATIONS
 * ========================================================================== */

/* Clear 8 cells at once */
static inline void inf_avx_clear_8(float* cells) {
    _mm256_storeu_ps(cells, _mm256_setzero_ps());
}

/* Clear entire map using AVX */
static inline void inf_avx_clear_map(float* map) {
    __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < INF_MAP_SIZE; i += 8) {
        _mm256_storeu_ps(&map[i], zero);
    }
}

/* Apply decay to 8 cells */
static inline void inf_avx_decay_8(float* cells, float decay_rate) {
    __m256 v = _mm256_loadu_ps(cells);
    __m256 d = _mm256_set1_ps(decay_rate);
    v = _mm256_mul_ps(v, d);
    _mm256_storeu_ps(cells, v);
}

/* Apply decay to entire map */
static inline void inf_avx_decay_map(float* map, float decay_rate) {
    __m256 d = _mm256_set1_ps(decay_rate);
    for (int i = 0; i < INF_MAP_SIZE; i += 8) {
        __m256 v = _mm256_loadu_ps(&map[i]);
        v = _mm256_mul_ps(v, d);
        _mm256_storeu_ps(&map[i], v);
    }
}

/* Add value to 8 cells */
static inline void inf_avx_add_8(float* cells, __m256 values) {
    __m256 current = _mm256_loadu_ps(cells);
    current = _mm256_add_ps(current, values);
    _mm256_storeu_ps(cells, current);
}

/* Blend two maps: out = a * weight + b * (1-weight) */
static inline void inf_avx_blend_maps(
    const float* map_a,
    const float* map_b,
    float weight,
    float* out
) {
    __m256 w = _mm256_set1_ps(weight);
    __m256 one_minus_w = _mm256_set1_ps(1.0f - weight);

    for (int i = 0; i < INF_MAP_SIZE; i += 8) {
        __m256 a = _mm256_loadu_ps(&map_a[i]);
        __m256 b = _mm256_loadu_ps(&map_b[i]);

        __m256 result = _mm256_add_ps(
            _mm256_mul_ps(a, w),
            _mm256_mul_ps(b, one_minus_w)
        );

        _mm256_storeu_ps(&out[i], result);
    }
}

/* ==========================================================================
 * THREAT INFLUENCE PROPAGATION
 * ========================================================================== */

/* Propagate threat from a single source position using AVX */
static inline void inf_avx_propagate_threat(
    float* threat_map,
    float source_x, float source_y,
    float strength,
    float falloff,
    int radius
) {
    int cx = (int)source_x;
    int cy = (int)source_y;

    /* Process in horizontal strips of 8 cells */
    for (int dy = -radius; dy <= radius; dy++) {
        int y = cy + dy;
        if (y < 0 || y >= INF_MAP_HEIGHT) continue;

        int row_start = y * INF_MAP_WIDTH;
        int x_start = cx - radius;
        int x_end = cx + radius;

        /* Clamp to map bounds */
        if (x_start < 0) x_start = 0;
        if (x_end >= INF_MAP_WIDTH) x_end = INF_MAP_WIDTH - 1;

        /* Process 8 cells at a time */
        for (int x = x_start; x <= x_end - 7; x += 8) {
            /* Calculate cell positions */
            __m256 cell_x = _mm256_set_ps(
                (float)(x+7), (float)(x+6), (float)(x+5), (float)(x+4),
                (float)(x+3), (float)(x+2), (float)(x+1), (float)x
            );
            __m256 cell_y = _mm256_set1_ps((float)y);
            __m256 sx = _mm256_set1_ps(source_x);
            __m256 sy = _mm256_set1_ps(source_y);

            /* Distance calculation */
            __m256 dx = _mm256_sub_ps(cell_x, sx);
            __m256 dy_v = _mm256_sub_ps(cell_y, sy);
            __m256 dist_sq = fmadd_ps(dy_v, dy_v, _mm256_mul_ps(dx, dx));
            __m256 dist = _mm256_sqrt_ps(dist_sq);

            /* Influence = strength * exp(-falloff * dist)
             * Approximate with: strength / (1 + falloff * dist) */
            __m256 fo = _mm256_set1_ps(falloff);
            __m256 str = _mm256_set1_ps(strength);
            __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f),
                                          _mm256_mul_ps(fo, dist));
            __m256 influence = _mm256_div_ps(str, denom);

            /* Only apply within radius */
            __m256 r = _mm256_set1_ps((float)radius);
            __m256 mask = _mm256_cmp_ps(dist, r, _CMP_LE_OQ);
            influence = _mm256_and_ps(influence, mask);

            /* Add to existing threat */
            int idx = row_start + x;
            __m256 current = _mm256_loadu_ps(&threat_map[idx]);
            current = _mm256_add_ps(current, influence);
            _mm256_storeu_ps(&threat_map[idx], current);
        }

        /* Handle remaining cells */
        for (int x = ((x_end - 7 - x_start) / 8) * 8 + x_start + 8; x <= x_end; x++) {
            if (x < 0 || x >= INF_MAP_WIDTH) continue;

            float dx = (float)x - source_x;
            float dy_f = (float)y - source_y;
            float dist = sqrtf(dx * dx + dy_f * dy_f);

            if (dist <= (float)radius) {
                float influence = strength / (1.0f + falloff * dist);
                threat_map[row_start + x] += influence;
            }
        }
    }
}

/* Update threat map from all entities using AVX */
static inline void inf_avx_update_threat(
    InfluenceMapAdvanced* inf,
    const GameState* game,
    int team_to_track    /* Which team's entities create threat (0=player, 1=enemy) */
) {
    /* Decay existing threat */
    inf_avx_decay_map(inf->threat, INFLUENCE_THREAT_DECAY);

    /* Add threat from each entity of the specified team */
    for (int i = 0; i < game->entity_count; i++) {
        const Entity* e = &game->entities[i];
        if (!e->alive || e->team != team_to_track) continue;

        /* Threat strength based on entity state */
        float strength = 1.0f;
        if (e->state >= STATE_ALERT) strength = 2.0f;
        if (e->state == STATE_COMBAT) strength = 3.0f;

        /* Add health factor */
        strength *= (e->health / e->max_health);

        /* Propagate threat */
        int radius = (int)(e->view_distance);
        inf_avx_propagate_threat(inf->threat, e->x, e->y, strength, 0.1f, radius);
    }
}

/* ==========================================================================
 * VISIBILITY CALCULATION
 * ========================================================================== */

/* Calculate visibility for 8 cells from single observer */
static inline __m256 inf_avx_visibility_8(
    __m256 cell_x, __m256 cell_y,
    float observer_x, float observer_y,
    float observer_facing_x, float observer_facing_y,
    float view_distance,
    float view_angle_cos      /* cos(half_view_angle) */
) {
    __m256 ox = _mm256_set1_ps(observer_x);
    __m256 oy = _mm256_set1_ps(observer_y);
    __m256 fx = _mm256_set1_ps(observer_facing_x);
    __m256 fy = _mm256_set1_ps(observer_facing_y);
    __m256 vd = _mm256_set1_ps(view_distance);
    __m256 va_cos = _mm256_set1_ps(view_angle_cos);

    /* Direction to cells */
    __m256 dx = _mm256_sub_ps(cell_x, ox);
    __m256 dy = _mm256_sub_ps(cell_y, oy);

    /* Distance */
    __m256 dist_sq = fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx));
    __m256 dist = _mm256_sqrt_ps(dist_sq);

    /* Normalize direction */
    __m256 inv_dist = _mm256_div_ps(_mm256_set1_ps(1.0f),
        _mm256_add_ps(dist, _mm256_set1_ps(0.001f)));
    __m256 norm_dx = _mm256_mul_ps(dx, inv_dist);
    __m256 norm_dy = _mm256_mul_ps(dy, inv_dist);

    /* Dot with facing */
    __m256 dot = fmadd_ps(norm_dy, fy, _mm256_mul_ps(norm_dx, fx));

    /* Visibility = (in_range AND in_cone) * (1 - dist/view_dist) */
    __m256 in_range = _mm256_cmp_ps(dist, vd, _CMP_LE_OQ);
    __m256 in_cone = _mm256_cmp_ps(dot, va_cos, _CMP_GE_OQ);
    __m256 valid = _mm256_and_ps(in_range, in_cone);

    /* Distance falloff */
    __m256 falloff = _mm256_sub_ps(_mm256_set1_ps(1.0f),
        _mm256_div_ps(dist, vd));
    falloff = _mm256_max_ps(falloff, _mm256_setzero_ps());

    return _mm256_and_ps(valid, falloff);
}

/* Update visibility map from all observers */
static inline void inf_avx_update_visibility(
    InfluenceMapAdvanced* inf,
    const GameState* game,
    int observing_team
) {
    /* Reset visibility */
    inf_avx_clear_map(inf->visibility);

    /* For each observer entity */
    for (int e = 0; e < game->entity_count; e++) {
        const Entity* obs = &game->entities[e];
        if (!obs->alive || obs->team != observing_team) continue;

        float facing_x = cosf(obs->facing_angle);
        float facing_y = sinf(obs->facing_angle);
        float view_cos = cosf(obs->view_cone_angle * 0.5f);
        int radius = (int)obs->view_distance;

        int cx = (int)obs->x;
        int cy = (int)obs->y;

        /* Process cells in batches of 8 */
        for (int dy = -radius; dy <= radius; dy++) {
            int y = cy + dy;
            if (y < 0 || y >= INF_MAP_HEIGHT) continue;

            int row = y * INF_MAP_WIDTH;

            for (int x = cx - radius; x <= cx + radius - 7; x += 8) {
                if (x < 0 || x + 7 >= INF_MAP_WIDTH) continue;

                __m256 cell_x = _mm256_set_ps(
                    (float)(x+7), (float)(x+6), (float)(x+5), (float)(x+4),
                    (float)(x+3), (float)(x+2), (float)(x+1), (float)x
                );
                __m256 cell_y = _mm256_set1_ps((float)y);

                __m256 vis = inf_avx_visibility_8(
                    cell_x, cell_y,
                    obs->x, obs->y,
                    facing_x, facing_y,
                    obs->view_distance,
                    view_cos
                );

                /* Max with existing visibility */
                int idx = row + x;
                __m256 current = _mm256_loadu_ps(&inf->visibility[idx]);
                current = _mm256_max_ps(current, vis);
                _mm256_storeu_ps(&inf->visibility[idx], current);
            }
        }
    }
}

/* ==========================================================================
 * COVER QUALITY CALCULATION
 * ========================================================================== */

/* Calculate cover quality for 8 cells based on surrounding walls */
static inline void inf_avx_calculate_cover_8(
    const uint8_t* tiles,
    int start_idx,
    __m256* cover_value
) {
    /* Count blocking neighbors for each cell */
    float __attribute__((aligned(32))) cover[8] = {0};

    for (int i = 0; i < 8; i++) {
        int idx = start_idx + i;
        int x = idx % INF_MAP_WIDTH;
        int y = idx / INF_MAP_WIDTH;

        if (x <= 0 || x >= INF_MAP_WIDTH - 1 ||
            y <= 0 || y >= INF_MAP_HEIGHT - 1) {
            cover[i] = 0;
            continue;
        }

        /* Check 8 neighbors */
        int blocking = 0;
        int neighbors[8] = {
            idx - INF_MAP_WIDTH - 1, idx - INF_MAP_WIDTH, idx - INF_MAP_WIDTH + 1,
            idx - 1, idx + 1,
            idx + INF_MAP_WIDTH - 1, idx + INF_MAP_WIDTH, idx + INF_MAP_WIDTH + 1
        };

        for (int n = 0; n < 8; n++) {
            uint8_t tile = tiles[neighbors[n]];
            if (tile == TILE_WALL || tile == TILE_COVER ||
                tile == TILE_CRATE || tile == TILE_BARREL ||
                tile == TILE_PILLAR || tile == TILE_TERMINAL) {
                blocking++;
            }
        }

        /* Cover value based on adjacent blocking tiles */
        cover[i] = (float)blocking / 8.0f;

        /* Bonus for actual cover tiles */
        uint8_t self_tile = tiles[idx];
        if (self_tile == TILE_COVER || self_tile == TILE_CRATE ||
            self_tile == TILE_BARREL || self_tile == TILE_TERMINAL) {
            cover[i] = 1.0f;
        }
    }

    *cover_value = _mm256_loadu_ps(cover);
}

/* Update entire cover map (expensive, do rarely) */
static inline void inf_avx_update_cover(
    InfluenceMapAdvanced* inf,
    const Level* level
) {
    /* Process map in batches of 8 */
    for (int i = 0; i < INF_MAP_SIZE; i += 8) {
        __m256 cover_val;
        inf_avx_calculate_cover_8(level->tiles, i, &cover_val);
        _mm256_storeu_ps(&inf->cover[i], cover_val);
    }
}

/* ==========================================================================
 * TACTICAL VALUE COMPUTATION
 * ========================================================================== */

/* Compute tactical value: combines all layers */
static inline void inf_avx_compute_tactical(
    InfluenceMapAdvanced* inf,
    float threat_weight,
    float visibility_weight,
    float cover_weight
) {
    __m256 tw = _mm256_set1_ps(threat_weight);
    __m256 vw = _mm256_set1_ps(visibility_weight);
    __m256 cw = _mm256_set1_ps(cover_weight);
    __m256 one = _mm256_set1_ps(1.0f);

    for (int i = 0; i < INF_MAP_SIZE; i += 8) {
        __m256 threat = _mm256_loadu_ps(&inf->threat[i]);
        __m256 vis = _mm256_loadu_ps(&inf->visibility[i]);
        __m256 cover = _mm256_loadu_ps(&inf->cover[i]);

        /* Tactical value = cover * cw - threat * tw - visibility * vw */
        __m256 tactical = _mm256_mul_ps(cover, cw);
        tactical = _mm256_sub_ps(tactical, _mm256_mul_ps(threat, tw));
        tactical = _mm256_sub_ps(tactical, _mm256_mul_ps(vis, vw));

        /* Clamp to reasonable range */
        tactical = _mm256_max_ps(tactical, _mm256_set1_ps(-10.0f));
        tactical = _mm256_min_ps(tactical, _mm256_set1_ps(10.0f));

        _mm256_storeu_ps(&inf->tactical[i], tactical);
    }
}

/* ==========================================================================
 * SOUND INFLUENCE
 * ========================================================================== */

/* Add sound event to influence map */
static inline void inf_avx_add_sound(
    InfluenceMapAdvanced* inf,
    float source_x, float source_y,
    float loudness,
    float corridor_mult
) {
    int radius = (int)(loudness * corridor_mult);
    inf_avx_propagate_threat(inf->sound, source_x, source_y,
                              loudness, 0.15f, radius);
}

/* Decay sound map (sounds fade faster than threat) */
static inline void inf_avx_decay_sound(InfluenceMapAdvanced* inf) {
    inf_avx_decay_map(inf->sound, 0.85f);
}

/* ==========================================================================
 * PATHFINDING COST COMPUTATION
 * ========================================================================== */

/* Compute path costs combining cover and threat */
static inline void inf_avx_compute_path_costs(
    InfluenceMapAdvanced* inf,
    const Level* level,
    float threat_avoidance,    /* How much to avoid threat (0-1) */
    float cover_preference     /* How much to prefer cover (0-1) */
) {
    __m256 ta = _mm256_set1_ps(threat_avoidance);
    __m256 cp = _mm256_set1_ps(cover_preference);
    __m256 base = _mm256_set1_ps(1.0f);

    for (int i = 0; i < INF_MAP_SIZE; i += 8) {
        __m256 threat = _mm256_loadu_ps(&inf->threat[i]);
        __m256 cover = _mm256_loadu_ps(&inf->cover[i]);

        /* Base cost = 1.0 + threat * avoidance - cover * preference */
        __m256 cost = _mm256_add_ps(base, _mm256_mul_ps(threat, ta));
        cost = _mm256_sub_ps(cost, _mm256_mul_ps(cover, cp));

        /* Minimum cost of 0.5 */
        cost = _mm256_max_ps(cost, _mm256_set1_ps(0.5f));

        _mm256_storeu_ps(&inf->path_cost[i], cost);
    }

    /* Mark impassable tiles as very high cost */
    for (int i = 0; i < INF_MAP_SIZE; i++) {
        uint8_t tile = level->tiles[i];
        if (tile == TILE_WALL || tile == TILE_WATER ||
            tile == TILE_PILLAR) {
            inf->path_cost[i] = 1000.0f;
        }
    }
}

/* ==========================================================================
 * MAP QUERIES
 * ========================================================================== */

/* Get value at position with bilinear interpolation using AVX */
static inline float inf_avx_sample(const float* map, float x, float y) {
    /* Clamp to bounds */
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= INF_MAP_WIDTH - 1) x = INF_MAP_WIDTH - 1.001f;
    if (y >= INF_MAP_HEIGHT - 1) y = INF_MAP_HEIGHT - 1.001f;

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = x - x0;
    float fy = y - y0;

    /* Gather 4 samples */
    float v00 = map[y0 * INF_MAP_WIDTH + x0];
    float v10 = map[y0 * INF_MAP_WIDTH + x1];
    float v01 = map[y1 * INF_MAP_WIDTH + x0];
    float v11 = map[y1 * INF_MAP_WIDTH + x1];

    /* Bilinear interpolation using SSE */
    __m128 samples = _mm_set_ps(v11, v01, v10, v00);
    __m128 weights = _mm_set_ps(
        fx * fy,               /* v11 */
        (1-fx) * fy,           /* v01 */
        fx * (1-fy),           /* v10 */
        (1-fx) * (1-fy)        /* v00 */
    );

    __m128 weighted = _mm_mul_ps(samples, weights);

    /* Horizontal sum */
    __m128 sum = _mm_add_ps(weighted, _mm_shuffle_ps(weighted, weighted, _MM_SHUFFLE(2,3,0,1)));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1,0,3,2)));

    return _mm_cvtss_f32(sum);
}

/* Sample 8 positions at once */
static inline void inf_avx_sample_8(
    const float* map,
    const float x[8], const float y[8],
    float out[8]
) {
    /* For simplicity, do 8 individual samples with SSE bilinear */
    for (int i = 0; i < 8; i++) {
        out[i] = inf_avx_sample(map, x[i], y[i]);
    }
}

/* Find best position in radius using AVX */
static inline void inf_avx_find_best_tactical(
    const InfluenceMapAdvanced* inf,
    float center_x, float center_y,
    float radius,
    float* best_x, float* best_y,
    float* best_value
) {
    int cx = (int)center_x;
    int cy = (int)center_y;
    int r = (int)radius;

    __m256 best_val = _mm256_set1_ps(-1e30f);
    __m256 best_px = _mm256_setzero_ps();
    __m256 best_py = _mm256_setzero_ps();

    for (int dy = -r; dy <= r; dy++) {
        int y = cy + dy;
        if (y < 0 || y >= INF_MAP_HEIGHT) continue;

        int row = y * INF_MAP_WIDTH;

        for (int x = cx - r; x <= cx + r - 7; x += 8) {
            if (x < 0 || x + 7 >= INF_MAP_WIDTH) continue;

            /* Load tactical values */
            __m256 val = _mm256_loadu_ps(&inf->tactical[row + x]);

            /* Cell positions */
            __m256 px = _mm256_set_ps(
                (float)(x+7), (float)(x+6), (float)(x+5), (float)(x+4),
                (float)(x+3), (float)(x+2), (float)(x+1), (float)x
            );
            __m256 py = _mm256_set1_ps((float)y);

            /* Check if within radius */
            __m256 dx = _mm256_sub_ps(px, _mm256_set1_ps(center_x));
            __m256 dy_v = _mm256_sub_ps(py, _mm256_set1_ps(center_y));
            __m256 dist_sq = fmadd_ps(dy_v, dy_v, _mm256_mul_ps(dx, dx));
            __m256 in_radius = _mm256_cmp_ps(dist_sq,
                _mm256_set1_ps(radius * radius), _CMP_LE_OQ);

            /* Update best if better and in radius */
            __m256 better = _mm256_cmp_ps(val, best_val, _CMP_GT_OQ);
            __m256 update = _mm256_and_ps(better, in_radius);

            best_val = _mm256_blendv_ps(best_val, val, update);
            best_px = _mm256_blendv_ps(best_px, px, update);
            best_py = _mm256_blendv_ps(best_py, py, update);
        }
    }

    /* Extract best from the 8 candidates */
    float __attribute__((aligned(32))) vals[8], pxs[8], pys[8];
    _mm256_storeu_ps(vals, best_val);
    _mm256_storeu_ps(pxs, best_px);
    _mm256_storeu_ps(pys, best_py);

    *best_value = vals[0];
    *best_x = pxs[0];
    *best_y = pys[0];

    for (int i = 1; i < 8; i++) {
        if (vals[i] > *best_value) {
            *best_value = vals[i];
            *best_x = pxs[i];
            *best_y = pys[i];
        }
    }
}

/* ==========================================================================
 * GRADIENT COMPUTATION
 * ========================================================================== */

/* Compute gradient (direction of steepest ascent/descent) at 8 positions */
static inline void inf_avx_gradient_8(
    const float* map,
    const float x[8], const float y[8],
    float grad_x[8], float grad_y[8]
) {
    float __attribute__((aligned(32))) samples_xp[8], samples_xn[8];
    float __attribute__((aligned(32))) samples_yp[8], samples_yn[8];

    /* Sample at offset positions */
    float delta = 0.5f;
    float x_plus[8], x_minus[8], y_plus[8], y_minus[8];

    for (int i = 0; i < 8; i++) {
        x_plus[i] = x[i] + delta;
        x_minus[i] = x[i] - delta;
        y_plus[i] = y[i] + delta;
        y_minus[i] = y[i] - delta;
    }

    inf_avx_sample_8(map, x_plus, y, samples_xp);
    inf_avx_sample_8(map, x_minus, y, samples_xn);
    inf_avx_sample_8(map, x, y_plus, samples_yp);
    inf_avx_sample_8(map, x, y_minus, samples_yn);

    /* Gradient = (f(x+d) - f(x-d)) / (2*d) */
    __m256 xp = _mm256_loadu_ps(samples_xp);
    __m256 xn = _mm256_loadu_ps(samples_xn);
    __m256 yp = _mm256_loadu_ps(samples_yp);
    __m256 yn = _mm256_loadu_ps(samples_yn);

    __m256 scale = _mm256_set1_ps(1.0f / (2.0f * delta));
    __m256 gx = _mm256_mul_ps(_mm256_sub_ps(xp, xn), scale);
    __m256 gy = _mm256_mul_ps(_mm256_sub_ps(yp, yn), scale);

    _mm256_storeu_ps(grad_x, gx);
    _mm256_storeu_ps(grad_y, gy);
}

/* ==========================================================================
 * INFLUENCE MAP API
 * ========================================================================== */

/* Initialize influence map */
void inf_init(InfluenceMapAdvanced* inf);

/* Full update (call once per many frames) */
void inf_full_update(InfluenceMapAdvanced* inf, const GameState* game);

/* Incremental update (call each frame) */
void inf_tick_update(InfluenceMapAdvanced* inf, const GameState* game);

/* Query functions */
float inf_get_threat(const InfluenceMapAdvanced* inf, float x, float y);
float inf_get_visibility(const InfluenceMapAdvanced* inf, float x, float y);
float inf_get_cover(const InfluenceMapAdvanced* inf, float x, float y);
float inf_get_tactical(const InfluenceMapAdvanced* inf, float x, float y);

/* Find safe position */
bool inf_find_safe_position(
    const InfluenceMapAdvanced* inf,
    float from_x, float from_y,
    float max_dist,
    float* safe_x, float* safe_y
);

/* Find flanking position */
bool inf_find_flank_position(
    const InfluenceMapAdvanced* inf,
    float from_x, float from_y,
    float target_x, float target_y,
    float* flank_x, float* flank_y
);

#endif /* SHOOTER_INFLUENCE_AVX_H */
