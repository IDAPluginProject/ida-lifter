/*
 * Advanced Player AI System
 * Sophisticated tactical AI for the player entity with predictive combat,
 * tactical planning, and extensive AVX/AVX2 usage.
 */

#include <immintrin.h>
#include <math.h>
#include <string.h>
#include "ai/player_ai.h"
#include "ai/prediction.h"
#include "ai/influence_avx.h"
#include "math/avx_ai_math.h"
#include "math/avx_math.h"
#include "config.h"
#include "core/rng.h"
#include "level/level.h"
#include "entity/entity.h"
#include "combat/bullet.h"
#include "math/sse_math.h"

/* ==========================================================================
 * PLAYER AI STATE
 * ========================================================================== */

typedef struct __attribute__((aligned(32))) {
    /* Movement prediction for all known enemies */
    MovementHistory enemy_histories[MAX_ENTITIES];

    /* Memory of enemy positions */
    AIMemory enemy_memory;

    /* Tactical planning */
    float best_cover_x, best_cover_y;
    float best_flank_x, best_flank_y;
    float retreat_direction_x, retreat_direction_y;

    /* Combat stats */
    int total_shots_fired;
    int total_hits;
    float running_accuracy;

    /* Prediction data */
    float predicted_aim_x, predicted_aim_y;
    float aim_confidence;

    /* Threat assessment cache */
    float threat_levels[MAX_ENTITIES];
    int priority_targets[8];
    int priority_count;

    /* Tactical state */
    int frames_since_last_kill;
    int frames_suppressed;
    bool is_being_flanked;
    int flank_direction;
} PlayerAIState;

static PlayerAIState player_ai_state;
static bool player_ai_initialized = false;

/* ==========================================================================
 * AVX ENEMY THREAT BATCH ASSESSMENT
 * ========================================================================== */

/* Assess threats from 8 enemies at once */
static void avx_assess_enemy_threats_8(
    const Entity* player,
    const Entity* enemies[8],
    const int* enemy_ids,
    int count,
    float* threat_levels
) {
    /* Load enemy positions */
    float __attribute__((aligned(32))) ex[8] = {0}, ey[8] = {0};
    float __attribute__((aligned(32))) evx[8] = {0}, evy[8] = {0};
    float __attribute__((aligned(32))) ehealth[8] = {0};
    float __attribute__((aligned(32))) efx[8] = {0}, efy[8] = {0};
    float __attribute__((aligned(32))) estate[8] = {0};

    for (int i = 0; i < count && i < 8; i++) {
        ex[i] = enemies[i]->x;
        ey[i] = enemies[i]->y;
        evx[i] = enemies[i]->vx;
        evy[i] = enemies[i]->vy;
        ehealth[i] = enemies[i]->health / enemies[i]->max_health;
        efx[i] = cosf(enemies[i]->facing_angle);
        efy[i] = sinf(enemies[i]->facing_angle);
        /* Give some threat value even to unaware enemies */
        estate[i] = (enemies[i]->state >= STATE_ALERT) ? 1.0f : 0.3f;
    }

    __m256 px = _mm256_set1_ps(player->x);
    __m256 py = _mm256_set1_ps(player->y);

    __m256 enemy_x = _mm256_loadu_ps(ex);
    __m256 enemy_y = _mm256_loadu_ps(ey);
    __m256 enemy_vx = _mm256_loadu_ps(evx);
    __m256 enemy_vy = _mm256_loadu_ps(evy);
    __m256 enemy_hp = _mm256_loadu_ps(ehealth);
    __m256 enemy_facing_x = _mm256_loadu_ps(efx);
    __m256 enemy_facing_y = _mm256_loadu_ps(efy);
    __m256 enemy_alert = _mm256_loadu_ps(estate);

    /* Distance calculation */
    __m256 dx = _mm256_sub_ps(px, enemy_x);
    __m256 dy = _mm256_sub_ps(py, enemy_y);
    __m256 dist_sq = fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx));
    __m256 dist = _mm256_sqrt_ps(dist_sq);
    __m256 inv_dist = _mm256_div_ps(_mm256_set1_ps(1.0f),
        _mm256_add_ps(dist, _mm256_set1_ps(0.001f)));

    /* How much enemy is facing player */
    __m256 norm_dx = _mm256_mul_ps(dx, inv_dist);
    __m256 norm_dy = _mm256_mul_ps(dy, inv_dist);
    __m256 facing_dot = fmadd_ps(norm_dx, enemy_facing_x,
        _mm256_mul_ps(norm_dy, enemy_facing_y));
    facing_dot = _mm256_max_ps(facing_dot, _mm256_setzero_ps());

    /* Enemy speed (approaching = more dangerous) */
    __m256 approach_speed = fmadd_ps(norm_dx, enemy_vx,
        _mm256_mul_ps(norm_dy, enemy_vy));
    approach_speed = _mm256_max_ps(approach_speed, _mm256_setzero_ps());

    /* Threat = (health * facing * alertness) * (50/dist) + approach_speed */
    __m256 threat = _mm256_mul_ps(enemy_hp, facing_dot);
    threat = _mm256_mul_ps(threat, enemy_alert);
    threat = _mm256_mul_ps(threat, _mm256_mul_ps(inv_dist, _mm256_set1_ps(50.0f)));
    threat = _mm256_add_ps(threat, _mm256_mul_ps(approach_speed, _mm256_set1_ps(5.0f)));

    /* Clamp */
    threat = _mm256_max_ps(threat, _mm256_setzero_ps());
    threat = _mm256_min_ps(threat, _mm256_set1_ps(100.0f));

    _mm256_storeu_ps(threat_levels, threat);
}

/* ==========================================================================
 * AVX TACTICAL POSITION EVALUATION
 * ========================================================================== */

/* Evaluate 8 potential positions for tactical value */
static void avx_evaluate_positions_8(
    float positions_x[8], float positions_y[8],
    const Entity* player,
    float threat_centroid_x, float threat_centroid_y,
    float* position_scores
) {
    __m256 px = _mm256_loadu_ps(positions_x);
    __m256 py = _mm256_loadu_ps(positions_y);
    __m256 player_x = _mm256_set1_ps(player->x);
    __m256 player_y = _mm256_set1_ps(player->y);
    __m256 threat_cx = _mm256_set1_ps(threat_centroid_x);
    __m256 threat_cy = _mm256_set1_ps(threat_centroid_y);

    /* Distance from player to positions (prefer closer) */
    __m256 dx_player = _mm256_sub_ps(px, player_x);
    __m256 dy_player = _mm256_sub_ps(py, player_y);
    __m256 dist_player = _mm256_sqrt_ps(fmadd_ps(dy_player, dy_player,
        _mm256_mul_ps(dx_player, dx_player)));

    /* Distance from threats to positions (prefer farther) */
    __m256 dx_threat = _mm256_sub_ps(px, threat_cx);
    __m256 dy_threat = _mm256_sub_ps(py, threat_cy);
    __m256 dist_threat = _mm256_sqrt_ps(fmadd_ps(dy_threat, dy_threat,
        _mm256_mul_ps(dx_threat, dx_threat)));

    /* Score = threat_distance / (player_distance + 1) */
    __m256 score = _mm256_div_ps(dist_threat,
        _mm256_add_ps(dist_player, _mm256_set1_ps(1.0f)));

    _mm256_storeu_ps(position_scores, score);
}

/* ==========================================================================
 * AVX DODGE DIRECTION CALCULATION
 * ========================================================================== */

/* Calculate best dodge direction from incoming fire */
static void avx_calculate_dodge(
    const Entity* player,
    const Bullet* bullets,
    int bullet_count,
    float* dodge_x, float* dodge_y
) {
    float __attribute__((aligned(32))) bx[8] = {0}, by[8] = {0};
    float __attribute__((aligned(32))) bvx[8] = {0}, bvy[8] = {0};
    float __attribute__((aligned(32))) active[8] = {0};

    /* Find bullets heading toward player */
    int found = 0;
    for (int i = 0; i < bullet_count && found < 8; i++) {
        const Bullet* b = &bullets[i];
        if (!b->active || b->team == 0) continue;

        /* Check if bullet heading toward player */
        float to_player_x = player->x - b->x;
        float to_player_y = player->y - b->y;
        float dot = to_player_x * b->vx + to_player_y * b->vy;

        if (dot > 0) {
            bx[found] = b->x;
            by[found] = b->y;
            bvx[found] = b->vx;
            bvy[found] = b->vy;
            active[found] = 1.0f;
            found++;
        }
    }

    if (found == 0) {
        *dodge_x = 0;
        *dodge_y = 0;
        return;
    }

    __m256 bullet_x = _mm256_loadu_ps(bx);
    __m256 bullet_y = _mm256_loadu_ps(by);
    __m256 bullet_vx = _mm256_loadu_ps(bvx);
    __m256 bullet_vy = _mm256_loadu_ps(bvy);
    __m256 bullet_active = _mm256_loadu_ps(active);

    pred_avx_dodge_direction_8(player->x, player->y,
        bullet_x, bullet_y, bullet_vx, bullet_vy, bullet_active,
        dodge_x, dodge_y);
}

/* ==========================================================================
 * AVX FLANKING DETECTION
 * ========================================================================== */

/* Detect if player is being flanked */
static void avx_detect_flanking(
    const Entity* player,
    const GameState* game,
    bool* is_flanked,
    int* flank_direction
) {
    /* Count enemies in each quadrant */
    int __attribute__((aligned(32))) quadrant_count[4] = {0, 0, 0, 0};

    float __attribute__((aligned(32))) ex[8], ey[8];
    int batch_count = 0;

    for (int i = 0; i < game->entity_count; i++) {
        const Entity* e = &game->entities[i];
        if (!e->alive || e->team == 0 || e->state < STATE_ALERT) continue;

        float dist = sse_distance(player->x, player->y, e->x, e->y);
        if (dist > 25.0f) continue;

        ex[batch_count] = e->x;
        ey[batch_count] = e->y;
        batch_count++;

        if (batch_count == 8) {
            /* Process batch */
            __m256 enemy_x = _mm256_loadu_ps(ex);
            __m256 enemy_y = _mm256_loadu_ps(ey);
            __m256 px = _mm256_set1_ps(player->x);
            __m256 py = _mm256_set1_ps(player->y);

            __m256 dx = _mm256_sub_ps(enemy_x, px);
            __m256 dy = _mm256_sub_ps(enemy_y, py);

            float __attribute__((aligned(32))) dx_arr[8], dy_arr[8];
            _mm256_storeu_ps(dx_arr, dx);
            _mm256_storeu_ps(dy_arr, dy);

            for (int j = 0; j < 8; j++) {
                int q;
                if (dx_arr[j] >= 0 && dy_arr[j] >= 0) q = 0;       /* NE */
                else if (dx_arr[j] < 0 && dy_arr[j] >= 0) q = 1;   /* NW */
                else if (dx_arr[j] < 0 && dy_arr[j] < 0) q = 2;    /* SW */
                else q = 3;                                         /* SE */
                quadrant_count[q]++;
            }

            batch_count = 0;
        }
    }

    /* Handle remaining */
    for (int j = 0; j < batch_count; j++) {
        float dx = ex[j] - player->x;
        float dy = ey[j] - player->y;
        int q;
        if (dx >= 0 && dy >= 0) q = 0;
        else if (dx < 0 && dy >= 0) q = 1;
        else if (dx < 0 && dy < 0) q = 2;
        else q = 3;
        quadrant_count[q]++;
    }

    /* Count occupied quadrants */
    int occupied = 0;
    int max_quadrant = 0;
    int max_count = 0;

    for (int i = 0; i < 4; i++) {
        if (quadrant_count[i] > 0) occupied++;
        if (quadrant_count[i] > max_count) {
            max_count = quadrant_count[i];
            max_quadrant = i;
        }
    }

    *is_flanked = (occupied >= 3);
    *flank_direction = max_quadrant;
}

/* ==========================================================================
 * AVX PRIORITY TARGET SELECTION
 * ========================================================================== */

/* Select priority targets using AVX sorting */
static void avx_select_priority_targets(
    const GameState* game,
    const Entity* player,
    PlayerAIState* ai
) {
    /* Calculate threat levels for all enemies */
    int enemy_ids[MAX_ENTITIES];
    int enemy_count = 0;

    for (int i = 0; i < game->entity_count; i++) {
        const Entity* e = &game->entities[i];
        if (e->alive && e->team != 0) {
            enemy_ids[enemy_count] = i;
            enemy_count++;
        }
    }

    /* Process in batches of 8 */
    for (int batch = 0; batch < enemy_count; batch += 8) {
        int batch_size = (enemy_count - batch < 8) ? (enemy_count - batch) : 8;

        const Entity* batch_enemies[8];
        for (int i = 0; i < batch_size; i++) {
            batch_enemies[i] = &game->entities[enemy_ids[batch + i]];
        }
        for (int i = batch_size; i < 8; i++) {
            batch_enemies[i] = &game->entities[0];  /* Placeholder */
        }

        float threats[8];
        avx_assess_enemy_threats_8(player, batch_enemies, &enemy_ids[batch],
            batch_size, threats);

        /* Store threat levels */
        for (int i = 0; i < batch_size; i++) {
            ai->threat_levels[enemy_ids[batch + i]] = threats[i];
        }
    }

    /* Find top 8 threats using AVX */
    ai->priority_count = 0;

    for (int p = 0; p < 8 && p < enemy_count; p++) {
        int best_id = -1;
        float best_threat = -1;

        for (int i = 0; i < enemy_count; i++) {
            int id = enemy_ids[i];
            bool already_selected = false;

            for (int j = 0; j < ai->priority_count; j++) {
                if (ai->priority_targets[j] == id) {
                    already_selected = true;
                    break;
                }
            }

            if (!already_selected && ai->threat_levels[id] > best_threat) {
                best_threat = ai->threat_levels[id];
                best_id = id;
            }
        }

        if (best_id >= 0) {
            ai->priority_targets[ai->priority_count++] = best_id;
        }
    }
}

/* ==========================================================================
 * AVX AIM PREDICTION
 * ========================================================================== */

static void avx_predict_aim(
    const Entity* player,
    const Entity* target,
    PlayerAIState* ai,
    float bullet_speed
) {
    MovementHistory* hist = &ai->enemy_histories[target->id];

    /* Update history */
    pred_history_add(hist, target->x, target->y, target->vx, target->vy);

    /* Calculate average velocity using AVX */
    float __attribute__((aligned(32))) vx_hist[8] = {0}, vy_hist[8] = {0};
    int samples = (hist->count < 8) ? hist->count : 8;

    for (int i = 0; i < samples; i++) {
        int idx = (hist->head - 1 - i + PRED_MAX_HISTORY) % PRED_MAX_HISTORY;
        vx_hist[i] = hist->vx[idx];
        vy_hist[i] = hist->vy[idx];
    }

    __m256 vx_vec = _mm256_loadu_ps(vx_hist);
    __m256 vy_vec = _mm256_loadu_ps(vy_hist);

    float avg_vx = hsum256_ps(vx_vec) / (samples > 0 ? samples : 1);
    float avg_vy = hsum256_ps(vy_vec) / (samples > 0 ? samples : 1);

    /* Calculate intercept */
    float intercept_t = pred_avx_intercept(
        player->x, player->y,
        target->x, target->y,
        avg_vx, avg_vy,
        bullet_speed,
        &ai->predicted_aim_x, &ai->predicted_aim_y
    );

    if (intercept_t > 0) {
        ai->aim_confidence = 0.9f - (intercept_t / 60.0f);
        if (ai->aim_confidence < 0.3f) ai->aim_confidence = 0.3f;
    } else {
        ai->predicted_aim_x = target->x;
        ai->predicted_aim_y = target->y;
        ai->aim_confidence = 0.5f;
    }
}

/* ==========================================================================
 * TACTICAL COVER SEARCH
 * ========================================================================== */

static void find_tactical_cover(
    const Entity* player,
    const GameState* game,
    float threat_cx, float threat_cy,
    float* cover_x, float* cover_y
) {
    /* Search for cover positions using AVX batch evaluation */
    float __attribute__((aligned(32))) candidate_x[8];
    float __attribute__((aligned(32))) candidate_y[8];
    float __attribute__((aligned(32))) candidate_scores[8];

    float best_score = -1e10f;
    *cover_x = player->x;
    *cover_y = player->y;

    /* Search in a grid around player */
    for (int search_y = -10; search_y <= 10; search_y += 3) {
        for (int search_x = -10; search_x <= 10; search_x += 8) {
            int batch_count = 0;

            for (int dx = 0; dx < 8 && search_x + dx <= 10; dx++) {
                int tx = (int)player->x + search_x + dx;
                int ty = (int)player->y + search_y;

                if (tx < 0 || tx >= LEVEL_WIDTH || ty < 0 || ty >= LEVEL_HEIGHT)
                    continue;

                if (!is_walkable(&game->level, tx, ty)) continue;

                /* Check if there's cover at this tile */
                uint8_t tile = game->level.tiles[ty * LEVEL_WIDTH + tx];
                if (tile != TILE_COVER && tile != TILE_CRATE &&
                    tile != TILE_BARREL && tile != TILE_TERMINAL) {
                    /* Check adjacent for cover */
                    bool has_adj_cover = false;
                    int adj_offsets[] = {-1, 0, 1, 0, 0, -1, 0, 1};
                    for (int a = 0; a < 4; a++) {
                        int ax = tx + adj_offsets[a*2];
                        int ay = ty + adj_offsets[a*2+1];
                        if (ax >= 0 && ax < LEVEL_WIDTH && ay >= 0 && ay < LEVEL_HEIGHT) {
                            uint8_t adj_tile = game->level.tiles[ay * LEVEL_WIDTH + ax];
                            if (adj_tile == TILE_COVER || adj_tile == TILE_CRATE ||
                                adj_tile == TILE_BARREL) {
                                has_adj_cover = true;
                                break;
                            }
                        }
                    }
                    if (!has_adj_cover) continue;
                }

                candidate_x[batch_count] = (float)tx + 0.5f;
                candidate_y[batch_count] = (float)ty + 0.5f;
                batch_count++;
            }

            if (batch_count > 0) {
                /* Pad remaining */
                for (int i = batch_count; i < 8; i++) {
                    candidate_x[i] = player->x;
                    candidate_y[i] = player->y;
                }

                avx_evaluate_positions_8(candidate_x, candidate_y, player,
                    threat_cx, threat_cy, candidate_scores);

                for (int i = 0; i < batch_count; i++) {
                    if (candidate_scores[i] > best_score) {
                        best_score = candidate_scores[i];
                        *cover_x = candidate_x[i];
                        *cover_y = candidate_y[i];
                    }
                }
            }
        }
    }
}

/* ==========================================================================
 * MAIN PLAYER AI UPDATE (ADVANCED)
 * ========================================================================== */

void update_player_ai_advanced(GameState* game) {
    Entity* player = &game->entities[game->player_id];
    if (!player->alive) return;

    /* Initialize if needed */
    if (!player_ai_initialized) {
        memset(&player_ai_state, 0, sizeof(player_ai_state));
        player_ai_initialized = true;
    }

    PlayerAIState* ai = &player_ai_state;
    WeaponStats wstats = weapon_get_stats(&player->weapon);

    /* Decrement timers */
    if (player->move_commit_timer > 0) player->move_commit_timer--;

    /* Update priority targets */
    avx_select_priority_targets(game, player, ai);

    /* Detect flanking */
    avx_detect_flanking(player, game, &ai->is_being_flanked, &ai->flank_direction);

    /* Calculate threat centroid using AVX */
    float __attribute__((aligned(32))) threat_x[8] = {0};
    float __attribute__((aligned(32))) threat_y[8] = {0};
    float __attribute__((aligned(32))) threat_weight[8] = {0};
    int threat_batch = 0;

    for (int i = 0; i < ai->priority_count && i < 8; i++) {
        const Entity* e = &game->entities[ai->priority_targets[i]];
        threat_x[threat_batch] = e->x;
        threat_y[threat_batch] = e->y;
        threat_weight[threat_batch] = ai->threat_levels[ai->priority_targets[i]];
        threat_batch++;
    }

    __m256 tx_vec = _mm256_loadu_ps(threat_x);
    __m256 ty_vec = _mm256_loadu_ps(threat_y);
    __m256 tw_vec = _mm256_loadu_ps(threat_weight);

    float total_weight = hsum256_ps(tw_vec);
    float threat_cx = (total_weight > 0) ?
        hsum256_ps(_mm256_mul_ps(tx_vec, tw_vec)) / total_weight : player->x;
    float threat_cy = (total_weight > 0) ?
        hsum256_ps(_mm256_mul_ps(ty_vec, tw_vec)) / total_weight : player->y;

    /* Check for nearby cover */
    player->has_cover_nearby = find_nearby_cover(&game->level, player->x, player->y,
        &player->cover_x, &player->cover_y);

    /* Calculate dodge direction from incoming bullets */
    float dodge_x = 0, dodge_y = 0;
    avx_calculate_dodge(player, game->bullets, game->bullet_count, &dodge_x, &dodge_y);

    /* Track damage */
    bool taking_fire = (player->damage_react_timer > 0);
    if (taking_fire) player->damage_react_timer--;

    /* Health management */
    float health_ratio = player->health / player->max_health;
    bool should_heal = (player->medpens > 0) &&
        ((health_ratio < 0.4f) ||
         (health_ratio < 0.7f && ai->priority_count == 0));

    /* Determine if we should retreat */
    bool should_retreat = (ai->is_being_flanked && health_ratio < 0.4f) ||
                          (ai->priority_count >= 4 && health_ratio < 0.5f) ||
                          (health_ratio < 0.2f && ai->priority_count >= 2);

    /* Movement variables */
    float move_x = 0, move_y = 0;
    bool should_shoot = false;
    int target_id = -1;

    /* Select primary target */
    if (ai->priority_count > 0) {
        target_id = ai->priority_targets[0];
        player->target_id = target_id;
    }

    /* State machine with advanced behaviors */
    switch (player->state) {
        case STATE_PATROL: {
            player->is_running = (ai->priority_count == 0);

            if (taking_fire) {
                if (should_retreat) {
                    player->state = STATE_RETREATING;
                } else if (player->has_cover_nearby) {
                    player->state = STATE_HIDING;
                } else {
                    player->state = STATE_COMBAT;
                }
                break;
            }

            if (ai->priority_count > 0) {
                if (should_retreat) {
                    player->state = STATE_RETREATING;
                } else {
                    player->state = STATE_COMBAT;
                }
            } else {
                /* Patrol with steering */
                if (player->patrol_x == 0 && player->patrol_y == 0) {
                    /* Pick new patrol target */
                    int room_idx = randi_range(&game->rng_state, 0,
                        game->level.room_count - 1);
                    Room* r = &game->level.rooms[room_idx];
                    player->patrol_x = r->x + r->width / 2;
                    player->patrol_y = r->y + r->height / 2;
                }

                float dx = player->patrol_x - player->x;
                float dy = player->patrol_y - player->y;
                float dist = sse_distance(0, 0, dx, dy);

                if (dist < 3.0f) {
                    player->patrol_x = 0;
                    player->patrol_y = 0;
                } else {
                    sse_normalize(&dx, &dy);
                    move_x = dx;
                    move_y = dy;
                    player->facing_angle = atan2f(dy, dx);
                }
            }
            break;
        }

        case STATE_COMBAT: {
            player->is_crouching = false;

            if (should_retreat) {
                player->state = STATE_RETREATING;
                break;
            }

            if (should_heal && player->has_cover_nearby) {
                player->state = STATE_HEALING;
                break;
            }

            if (target_id >= 0) {
                Entity* target = &game->entities[target_id];

                if (!target->alive) {
                    ai->frames_since_last_kill = 0;
                    player->target_id = -1;
                    if (ai->priority_count <= 1) {
                        player->state = STATE_PATROL;
                    }
                    break;
                }

                /* Calculate predicted aim */
                avx_predict_aim(player, target, ai, BULLET_SPEED);

                /* Check visibility */
                int vis = check_view_cone(&game->level, player->x, player->y,
                    player->facing_angle, target->x, target->y,
                    player->view_cone_angle, player->view_distance, true);

                float tdist = sse_distance(player->x, player->y, target->x, target->y);

                if (vis > 0) {
                    player->frames_target_visible++;

                    /* Face predicted position */
                    float aim_dx = ai->predicted_aim_x - player->x;
                    float aim_dy = ai->predicted_aim_y - player->y;
                    player->facing_angle = atan2f(aim_dy, aim_dx);

                    float ideal_range = wstats.range * 0.5f;

                    /* Movement with dodge integration */
                    float combat_move_x = 0, combat_move_y = 0;

                    if (tdist > ideal_range + 5.0f) {
                        /* Advance */
                        float dx = target->x - player->x;
                        float dy = target->y - player->y;
                        sse_normalize(&dx, &dy);
                        combat_move_x = dx;
                        combat_move_y = dy;
                    } else if (tdist < ideal_range - 3.0f) {
                        /* Back up */
                        float dx = player->x - target->x;
                        float dy = player->y - target->y;
                        sse_normalize(&dx, &dy);
                        combat_move_x = dx * 0.7f;
                        combat_move_y = dy * 0.7f;
                    } else {
                        /* Strafe */
                        float dx = target->x - player->x;
                        float dy = target->y - player->y;
                        sse_normalize(&dx, &dy);
                        float strafe = ((game->frame / 60) % 2 == 0) ? 1.0f : -1.0f;
                        combat_move_x = -dy * strafe * 0.5f;
                        combat_move_y = dx * strafe * 0.5f;
                    }

                    /* Blend with dodge */
                    float dodge_strength = (dodge_x != 0 || dodge_y != 0) ? 0.7f : 0.0f;
                    move_x = combat_move_x * (1.0f - dodge_strength) + dodge_x * dodge_strength;
                    move_y = combat_move_y * (1.0f - dodge_strength) + dodge_y * dodge_strength;

                    /* Shoot with prediction */
                    if (tdist < wstats.range && player->frames_target_visible > 5) {
                        if (ai->aim_confidence > 0.4f || tdist < 10.0f) {
                            should_shoot = true;
                        }
                    }
                } else {
                    /* Lost sight - hunt */
                    player->frames_target_visible = 0;
                    player->last_seen_x = target->x;
                    player->last_seen_y = target->y;

                    float dx = player->last_seen_x - player->x;
                    float dy = player->last_seen_y - player->y;
                    float dist = sse_distance(0, 0, dx, dy);

                    if (dist > 2.0f) {
                        sse_normalize(&dx, &dy);
                        move_x = dx;
                        move_y = dy;
                        player->is_running = true;
                    }
                }
            }

            /* Auto-reload when low */
            if (player->weapon.mag_current <= 2 && player->weapon.reserve > 0) {
                player->reload_timer = wstats.reload_time;
            }
            break;
        }

        case STATE_RETREATING: {
            player->is_running = true;

            /* Find tactical retreat position */
            find_tactical_cover(player, game, threat_cx, threat_cy,
                &ai->best_cover_x, &ai->best_cover_y);

            float dx = ai->best_cover_x - player->x;
            float dy = ai->best_cover_y - player->y;
            float dist = sse_distance(0, 0, dx, dy);

            if (dist > 1.0f) {
                sse_normalize(&dx, &dy);
                move_x = dx;
                move_y = dy;

                /* Face enemies while retreating */
                if (target_id >= 0) {
                    Entity* target = &game->entities[target_id];
                    player->facing_angle = atan2f(target->y - player->y,
                                                   target->x - player->x);

                    /* Shoot while retreating */
                    float tdist = sse_distance(player->x, player->y, target->x, target->y);
                    if (tdist < wstats.range) {
                        int vis = check_view_cone(&game->level, player->x, player->y,
                            player->facing_angle, target->x, target->y, PI, 50.0f, true);
                        if (vis > 0) should_shoot = true;
                    }
                }
            } else {
                /* Reached cover */
                player->state = STATE_HIDING;
                player->alert_timer = 30;
            }

            /* Stop retreating if situation improves */
            if (!should_retreat) {
                player->state = STATE_COMBAT;
            }
            break;
        }

        case STATE_HIDING: {
            player->is_crouching = true;
            player->alert_timer--;

            /* Move to cover */
            if (player->has_cover_nearby) {
                float dx = player->cover_x - player->x;
                float dy = player->cover_y - player->y;
                float dist = sse_distance(0, 0, dx, dy);

                if (dist > 1.0f) {
                    sse_normalize(&dx, &dy);
                    move_x = dx * 0.5f;
                    move_y = dy * 0.5f;
                }
            }

            /* Track target */
            if (target_id >= 0) {
                Entity* target = &game->entities[target_id];
                float dx = target->x - player->x;
                float dy = target->y - player->y;
                float target_angle = atan2f(dy, dx);

                /* Smooth turning */
                float angle_diff = target_angle - player->facing_angle;
                while (angle_diff > PI) angle_diff -= 2*PI;
                while (angle_diff < -PI) angle_diff += 2*PI;
                player->facing_angle += angle_diff * 0.1f;
            }

            /* Transition conditions */
            if (player->alert_timer <= 0) {
                player->state = STATE_COMBAT;
                player->is_crouching = false;
            }

            if (should_heal) {
                player->state = STATE_HEALING;
            }

            if (taking_fire && should_retreat) {
                player->state = STATE_RETREATING;
            }
            break;
        }

        case STATE_HEALING: {
            player->is_crouching = true;

            /* Move to cover while healing */
            if (player->has_cover_nearby) {
                float dx = player->cover_x - player->x;
                float dy = player->cover_y - player->y;
                float dist = sse_distance(0, 0, dx, dy);

                if (dist > 1.5f) {
                    sse_normalize(&dx, &dy);
                    move_x = dx * 0.3f;
                    move_y = dy * 0.3f;
                } else {
                    move_x = 0;
                    move_y = 0;

                    if (player->healing_timer == 0) {
                        player->healing_timer = MEDPEN_USE_TIME;
                    }

                    player->healing_timer--;

                    if (player->healing_timer <= 0) {
                        player->medpens--;
                        player->health += MEDPEN_HEAL_AMOUNT;
                        if (player->health > PLAYER_MAX_HEALTH)
                            player->health = PLAYER_MAX_HEALTH;
                        player->state = (ai->priority_count > 0) ? STATE_COMBAT : STATE_PATROL;
                    }
                }
            } else {
                /* Heal anyway */
                if (player->healing_timer == 0) {
                    player->healing_timer = MEDPEN_USE_TIME;
                }

                player->healing_timer--;

                if (player->healing_timer <= 0) {
                    player->medpens--;
                    player->health += MEDPEN_HEAL_AMOUNT;
                    if (player->health > PLAYER_MAX_HEALTH)
                        player->health = PLAYER_MAX_HEALTH;
                    player->state = (ai->priority_count > 0) ? STATE_COMBAT : STATE_PATROL;
                }
            }

            /* Interrupt if taking fire */
            if (taking_fire && player->healing_timer > 0) {
                player->healing_timer = 0;
                player->state = should_retreat ? STATE_RETREATING : STATE_COMBAT;
            }
            break;
        }

        case STATE_RELOAD: {
            player->reload_timer--;
            move_x = 0;
            move_y = 0;

            if (player->reload_timer <= 0) {
                int to_load = wstats.mag_size;
                if (player->weapon.reserve < to_load) to_load = player->weapon.reserve;
                player->weapon.mag_current = to_load;
                player->weapon.reserve -= to_load;
                player->state = (ai->priority_count > 0) ? STATE_COMBAT : STATE_PATROL;
            }
            break;
        }

        default:
            player->state = STATE_PATROL;
            break;
    }

    /* Execute shooting with prediction */
    if (should_shoot && player->fire_cooldown <= 0 && player->reload_timer <= 0 &&
        player->weapon.mag_current > 0) {

        /* Use predicted aim angle */
        float aim_angle = atan2f(ai->predicted_aim_y - player->y,
                                  ai->predicted_aim_x - player->x);

        spawn_bullet(game, game->player_id, player->x, player->y, aim_angle, wstats.accuracy);
        player->weapon.mag_current--;
        player->fire_cooldown = wstats.fire_rate;

        ai->total_shots_fired++;
    }

    /* Apply movement with pathfinding */
    float speed = player->is_running ? PLAYER_RUN_SPEED :
                  (player->is_crouching ? PLAYER_WALK_SPEED * 0.5f : PLAYER_WALK_SPEED);

    /* Stamina */
    if (player->is_running) {
        player->stamina -= STAMINA_DRAIN_RATE;
        if (player->stamina <= 0) {
            player->stamina = 0;
            player->is_running = false;
        }
    } else {
        player->stamina += STAMINA_REGEN_RATE;
        if (player->stamina > PLAYER_MAX_STAMINA) player->stamina = PLAYER_MAX_STAMINA;
    }

    if (move_x != 0 || move_y != 0) {
        float target_x = player->x + move_x * 10.0f;
        float target_y = player->y + move_y * 10.0f;
        float next_x, next_y;

        if (find_path(&game->level, player->x, player->y, target_x, target_y,
                      &next_x, &next_y)) {
            player->vx = (next_x - player->x) * speed;
            player->vy = (next_y - player->y) * speed;
        }
    } else {
        /* Apply friction using AVX */
        __m128 vel = _mm_set_ps(0, 0, player->vy, player->vx);
        __m128 friction = _mm_set1_ps(0.5f);
        vel = _mm_mul_ps(vel, friction);

        float __attribute__((aligned(16))) v[4];
        _mm_store_ps(v, vel);
        player->vx = v[0];
        player->vy = v[1];
    }

    /* Handle reload */
    if (player->reload_timer > 0 && player->state != STATE_RELOAD) {
        player->reload_timer--;
        if (player->reload_timer <= 0) {
            int to_load = wstats.mag_size;
            if (player->weapon.reserve < to_load) to_load = player->weapon.reserve;
            player->weapon.mag_current = to_load;
            player->weapon.reserve -= to_load;
        }
    }

    player->fire_cooldown--;
    ai->frames_since_last_kill++;
}

/* ==========================================================================
 * INITIALIZATION AND SHUTDOWN
 * ========================================================================== */

void init_player_ai_advanced(GameState* game) {
    (void)game;  /* May be used for future initialization */

    /* Clear player AI state */
    memset(&player_ai_state, 0, sizeof(player_ai_state));

    /* Initialize default values */
    player_ai_state.aim_confidence = 0.5f;
    player_ai_state.running_accuracy = 0.5f;

    player_ai_initialized = true;
}

void shutdown_player_ai_advanced(void) {
    /* Clear player AI state */
    memset(&player_ai_state, 0, sizeof(player_ai_state));
}
