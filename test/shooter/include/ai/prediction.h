/*
 * Prediction System with AVX Trajectory Calculations
 * Advanced prediction for target movement, projectile interception, and threat anticipation.
 * Heavy AVX/AVX2 usage for trajectory and ballistic calculations.
 */

#ifndef SHOOTER_PREDICTION_H
#define SHOOTER_PREDICTION_H

#include <immintrin.h>
#include <math.h>
#include "../config.h"
#include "../types.h"
#include "../math/avx_math.h"

/* ==========================================================================
 * PREDICTION CONSTANTS
 * ========================================================================== */

#define PRED_MAX_HISTORY 16       /* Frames of position history */
#define PRED_LOOKAHEAD_FRAMES 30  /* How far to predict */
#define PRED_SAMPLES 8            /* Sample points for trajectory */

/* ==========================================================================
 * MOVEMENT HISTORY TRACKING
 * ========================================================================== */

typedef struct __attribute__((aligned(32))) {
    float x[PRED_MAX_HISTORY];
    float y[PRED_MAX_HISTORY];
    float vx[PRED_MAX_HISTORY];
    float vy[PRED_MAX_HISTORY];
    float ax[PRED_MAX_HISTORY];   /* Acceleration (computed) */
    float ay[PRED_MAX_HISTORY];
    int head;                      /* Circular buffer head */
    int count;                     /* Valid entries */
} MovementHistory;

/* Add observation to history */
static inline void pred_history_add(MovementHistory* hist, float x, float y,
                                     float vx, float vy) {
    /* Compute acceleration from velocity delta */
    if (hist->count > 0) {
        int prev = (hist->head - 1 + PRED_MAX_HISTORY) % PRED_MAX_HISTORY;
        hist->ax[hist->head] = vx - hist->vx[prev];
        hist->ay[hist->head] = vy - hist->vy[prev];
    } else {
        hist->ax[hist->head] = 0;
        hist->ay[hist->head] = 0;
    }

    hist->x[hist->head] = x;
    hist->y[hist->head] = y;
    hist->vx[hist->head] = vx;
    hist->vy[hist->head] = vy;

    hist->head = (hist->head + 1) % PRED_MAX_HISTORY;
    if (hist->count < PRED_MAX_HISTORY) hist->count++;
}

/* ==========================================================================
 * AVX TRAJECTORY PREDICTION
 * ========================================================================== */

/* Predict position at multiple future times using AVX */
static inline void pred_avx_trajectory_8(
    float x0, float y0,
    float vx, float vy,
    float ax, float ay,
    const float times[8],      /* 8 time points to predict */
    float out_x[8], float out_y[8]
) {
    __m256 t = _mm256_loadu_ps(times);
    __m256 t2 = _mm256_mul_ps(t, t);
    __m256 half = _mm256_set1_ps(0.5f);

    __m256 px0 = _mm256_set1_ps(x0);
    __m256 py0 = _mm256_set1_ps(y0);
    __m256 pvx = _mm256_set1_ps(vx);
    __m256 pvy = _mm256_set1_ps(vy);
    __m256 pax = _mm256_set1_ps(ax);
    __m256 pay = _mm256_set1_ps(ay);

    /* x(t) = x0 + vx*t + 0.5*ax*t^2 */
    __m256 pred_x = fmadd_ps(pax, _mm256_mul_ps(half, t2),
                              fmadd_ps(pvx, t, px0));
    __m256 pred_y = fmadd_ps(pay, _mm256_mul_ps(half, t2),
                              fmadd_ps(pvy, t, py0));

    _mm256_storeu_ps(out_x, pred_x);
    _mm256_storeu_ps(out_y, pred_y);
}

/* Predict velocity at multiple future times */
static inline void pred_avx_velocity_8(
    float vx0, float vy0,
    float ax, float ay,
    const float times[8],
    float out_vx[8], float out_vy[8]
) {
    __m256 t = _mm256_loadu_ps(times);
    __m256 pvx = _mm256_set1_ps(vx0);
    __m256 pvy = _mm256_set1_ps(vy0);
    __m256 pax = _mm256_set1_ps(ax);
    __m256 pay = _mm256_set1_ps(ay);

    /* v(t) = v0 + a*t */
    __m256 pred_vx = fmadd_ps(pax, t, pvx);
    __m256 pred_vy = fmadd_ps(pay, t, pvy);

    _mm256_storeu_ps(out_vx, pred_vx);
    _mm256_storeu_ps(out_vy, pred_vy);
}

/* ==========================================================================
 * AVX INTERCEPT CALCULATION
 * ========================================================================== */

/* Calculate intercept point for a projectile
 * Returns time to intercept (negative if impossible) */
static inline float pred_avx_intercept(
    float shooter_x, float shooter_y,
    float target_x, float target_y,
    float target_vx, float target_vy,
    float bullet_speed,
    float* aim_x, float* aim_y
) {
    /* Vector from shooter to target */
    float dx = target_x - shooter_x;
    float dy = target_y - shooter_y;

    /* Quadratic equation: |P + V*t|^2 = (bullet_speed * t)^2
     * where P = initial offset, V = target velocity
     *
     * Expanding: (P.x + V.x*t)^2 + (P.y + V.y*t)^2 = bs^2 * t^2
     * (V.x^2 + V.y^2 - bs^2) * t^2 + 2*(P.x*V.x + P.y*V.y) * t + (P.x^2 + P.y^2) = 0
     */

    float a = target_vx * target_vx + target_vy * target_vy -
              bullet_speed * bullet_speed;
    float b = 2.0f * (dx * target_vx + dy * target_vy);
    float c = dx * dx + dy * dy;

    /* Use AVX for discriminant and roots */
    __m256 va = _mm256_set1_ps(a);
    __m256 vb = _mm256_set1_ps(b);
    __m256 vc = _mm256_set1_ps(c);
    __m256 v4 = _mm256_set1_ps(4.0f);
    __m256 v2 = _mm256_set1_ps(2.0f);

    /* discriminant = b^2 - 4ac */
    __m256 disc = _mm256_sub_ps(
        _mm256_mul_ps(vb, vb),
        _mm256_mul_ps(v4, _mm256_mul_ps(va, vc))
    );

    float discriminant = _mm_cvtss_f32(_mm256_castps256_ps128(disc));

    if (discriminant < 0) {
        /* No intercept possible - just aim at current position */
        *aim_x = target_x;
        *aim_y = target_y;
        return -1.0f;
    }

    float sqrt_disc = sqrtf(discriminant);

    /* Two solutions: t = (-b +/- sqrt(disc)) / (2a) */
    float t1 = (-b + sqrt_disc) / (2.0f * a);
    float t2 = (-b - sqrt_disc) / (2.0f * a);

    /* Choose smallest positive time */
    float t = -1.0f;
    if (t1 > 0 && (t1 < t2 || t2 <= 0)) t = t1;
    else if (t2 > 0) t = t2;

    if (t > 0) {
        /* Calculate intercept point */
        *aim_x = target_x + target_vx * t;
        *aim_y = target_y + target_vy * t;
        return t;
    } else {
        *aim_x = target_x;
        *aim_y = target_y;
        return -1.0f;
    }
}

/* Calculate intercepts for 8 shooters against one target */
static inline void pred_avx_intercept_8(
    __m256 shooter_x, __m256 shooter_y,
    float target_x, float target_y,
    float target_vx, float target_vy,
    float bullet_speed,
    __m256* aim_x, __m256* aim_y,
    __m256* intercept_time
) {
    __m256 tx = _mm256_set1_ps(target_x);
    __m256 ty = _mm256_set1_ps(target_y);
    __m256 tvx = _mm256_set1_ps(target_vx);
    __m256 tvy = _mm256_set1_ps(target_vy);
    __m256 bs = _mm256_set1_ps(bullet_speed);
    __m256 bs2 = _mm256_mul_ps(bs, bs);

    /* Vector from each shooter to target */
    __m256 dx = _mm256_sub_ps(tx, shooter_x);
    __m256 dy = _mm256_sub_ps(ty, shooter_y);

    /* Quadratic coefficients */
    __m256 a = _mm256_sub_ps(
        _mm256_add_ps(_mm256_mul_ps(tvx, tvx), _mm256_mul_ps(tvy, tvy)),
        bs2
    );
    __m256 b = _mm256_mul_ps(_mm256_set1_ps(2.0f),
        _mm256_add_ps(_mm256_mul_ps(dx, tvx), _mm256_mul_ps(dy, tvy)));
    __m256 c = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));

    /* Discriminant */
    __m256 four = _mm256_set1_ps(4.0f);
    __m256 disc = _mm256_sub_ps(
        _mm256_mul_ps(b, b),
        _mm256_mul_ps(four, _mm256_mul_ps(a, c))
    );

    /* sqrt(disc) where disc >= 0 */
    __m256 valid = _mm256_cmp_ps(disc, _mm256_setzero_ps(), _CMP_GE_OQ);
    __m256 sqrt_disc = _mm256_sqrt_ps(_mm256_max_ps(disc, _mm256_setzero_ps()));

    /* t = (-b - sqrt(disc)) / (2a) - prefer smaller positive root */
    __m256 two_a = _mm256_mul_ps(_mm256_set1_ps(2.0f), a);
    __m256 neg_b = _mm256_sub_ps(_mm256_setzero_ps(), b);

    __m256 t1 = _mm256_div_ps(_mm256_add_ps(neg_b, sqrt_disc), two_a);
    __m256 t2 = _mm256_div_ps(_mm256_sub_ps(neg_b, sqrt_disc), two_a);

    /* Choose smallest positive */
    __m256 zero = _mm256_setzero_ps();
    __m256 t1_pos = _mm256_cmp_ps(t1, zero, _CMP_GT_OQ);
    __m256 t2_pos = _mm256_cmp_ps(t2, zero, _CMP_GT_OQ);
    __m256 t2_better = _mm256_and_ps(t2_pos,
        _mm256_or_ps(_mm256_cmp_ps(t2, t1, _CMP_LT_OQ),
                     _mm256_cmp_ps(t1, zero, _CMP_LE_OQ)));

    __m256 t = _mm256_blendv_ps(
        _mm256_blendv_ps(_mm256_set1_ps(-1.0f), t1, t1_pos),
        t2,
        t2_better
    );

    /* Invalid if no solution or t <= 0 */
    t = _mm256_and_ps(t, valid);
    __m256 has_solution = _mm256_cmp_ps(t, zero, _CMP_GT_OQ);

    /* Calculate intercept points */
    __m256 pred_x = fmadd_ps(tvx, t, tx);
    __m256 pred_y = fmadd_ps(tvy, t, ty);

    /* Use current position if no solution */
    *aim_x = _mm256_blendv_ps(tx, pred_x, has_solution);
    *aim_y = _mm256_blendv_ps(ty, pred_y, has_solution);
    *intercept_time = t;
}

/* ==========================================================================
 * MOVEMENT PATTERN ANALYSIS
 * ========================================================================== */

typedef enum {
    PATTERN_STATIONARY = 0,
    PATTERN_LINEAR,
    PATTERN_CIRCULAR,
    PATTERN_ZIGZAG,
    PATTERN_ERRATIC,
    PATTERN_COVER_SEEKING,
    PATTERN_AGGRESSIVE,
    PATTERN_RETREATING
} MovementPattern;

/* Analyze movement history to determine pattern using AVX */
static inline MovementPattern pred_avx_analyze_pattern(const MovementHistory* hist) {
    if (hist->count < 4) return PATTERN_STATIONARY;

    /* Load recent positions into AVX registers */
    float __attribute__((aligned(32))) recent_x[8];
    float __attribute__((aligned(32))) recent_y[8];
    float __attribute__((aligned(32))) recent_vx[8];
    float __attribute__((aligned(32))) recent_vy[8];

    int samples = (hist->count < 8) ? hist->count : 8;
    for (int i = 0; i < samples; i++) {
        int idx = (hist->head - 1 - i + PRED_MAX_HISTORY) % PRED_MAX_HISTORY;
        recent_x[i] = hist->x[idx];
        recent_y[i] = hist->y[idx];
        recent_vx[i] = hist->vx[idx];
        recent_vy[i] = hist->vy[idx];
    }

    __m256 vx = _mm256_loadu_ps(recent_vx);
    __m256 vy = _mm256_loadu_ps(recent_vy);

    /* Calculate speed */
    __m256 speed_sq = fmadd_ps(vy, vy, _mm256_mul_ps(vx, vx));
    __m256 speed = _mm256_sqrt_ps(speed_sq);

    /* Check if mostly stationary */
    __m256 low_speed = _mm256_cmp_ps(speed, _mm256_set1_ps(0.05f), _CMP_LT_OQ);
    int stationary_mask = _mm256_movemask_ps(low_speed);
    if (__builtin_popcount(stationary_mask) >= 6) {
        return PATTERN_STATIONARY;
    }

    /* Calculate heading changes */
    float heading_changes = 0;
    for (int i = 1; i < samples; i++) {
        float dot = recent_vx[i-1] * recent_vx[i] + recent_vy[i-1] * recent_vy[i];
        float len1 = sqrtf(recent_vx[i-1] * recent_vx[i-1] + recent_vy[i-1] * recent_vy[i-1]);
        float len2 = sqrtf(recent_vx[i] * recent_vx[i] + recent_vy[i] * recent_vy[i]);
        if (len1 > 0.01f && len2 > 0.01f) {
            float cos_angle = dot / (len1 * len2);
            if (cos_angle < 0.8f) heading_changes += 1.0f;
        }
    }

    /* High heading changes = zigzag or erratic */
    if (heading_changes >= 4) {
        return PATTERN_ZIGZAG;
    } else if (heading_changes >= 2) {
        return PATTERN_ERRATIC;
    }

    /* Check for linear movement */
    __m256 mean_vx = _mm256_set1_ps(hsum256_ps(vx) / samples);
    __m256 mean_vy = _mm256_set1_ps(hsum256_ps(vy) / samples);

    __m256 var_vx = _mm256_sub_ps(vx, mean_vx);
    __m256 var_vy = _mm256_sub_ps(vy, mean_vy);
    var_vx = _mm256_mul_ps(var_vx, var_vx);
    var_vy = _mm256_mul_ps(var_vy, var_vy);

    float variance = (hsum256_ps(var_vx) + hsum256_ps(var_vy)) / (2 * samples);
    if (variance < 0.01f) {
        return PATTERN_LINEAR;
    }

    /* Check movement direction relative to some reference */
    float avg_vx = hsum256_ps(vx) / samples;
    float avg_vy = hsum256_ps(vy) / samples;
    float avg_speed = sqrtf(avg_vx * avg_vx + avg_vy * avg_vy);

    if (avg_speed > PLAYER_RUN_SPEED * 0.8f) {
        return PATTERN_AGGRESSIVE;
    }

    return PATTERN_LINEAR;
}

/* ==========================================================================
 * THREAT PREDICTION
 * ========================================================================== */

typedef struct __attribute__((aligned(32))) {
    float predicted_x;
    float predicted_y;
    float time_to_arrive;
    float danger_level;
    MovementPattern pattern;
} ThreatPrediction;

/* Predict future threat positions for 8 threats */
static inline void pred_avx_threat_forecast_8(
    const Entity* threats[8],
    const MovementHistory* histories[8],
    float forecast_time,
    ThreatPrediction predictions[8]
) {
    /* Load current positions */
    __m256 x = _mm256_set_ps(
        threats[7]->x, threats[6]->x, threats[5]->x, threats[4]->x,
        threats[3]->x, threats[2]->x, threats[1]->x, threats[0]->x
    );
    __m256 y = _mm256_set_ps(
        threats[7]->y, threats[6]->y, threats[5]->y, threats[4]->y,
        threats[3]->y, threats[2]->y, threats[1]->y, threats[0]->y
    );
    __m256 vx = _mm256_set_ps(
        threats[7]->vx, threats[6]->vx, threats[5]->vx, threats[4]->vx,
        threats[3]->vx, threats[2]->vx, threats[1]->vx, threats[0]->vx
    );
    __m256 vy = _mm256_set_ps(
        threats[7]->vy, threats[6]->vy, threats[5]->vy, threats[4]->vy,
        threats[3]->vy, threats[2]->vy, threats[1]->vy, threats[0]->vy
    );

    /* Estimate acceleration from history */
    float __attribute__((aligned(32))) ax_arr[8] = {0};
    float __attribute__((aligned(32))) ay_arr[8] = {0};

    for (int i = 0; i < 8; i++) {
        if (histories[i] && histories[i]->count > 1) {
            int latest = (histories[i]->head - 1 + PRED_MAX_HISTORY) % PRED_MAX_HISTORY;
            ax_arr[i] = histories[i]->ax[latest];
            ay_arr[i] = histories[i]->ay[latest];
        }
    }

    __m256 ax = _mm256_loadu_ps(ax_arr);
    __m256 ay = _mm256_loadu_ps(ay_arr);
    __m256 t = _mm256_set1_ps(forecast_time);
    __m256 t2 = _mm256_mul_ps(t, t);
    __m256 half = _mm256_set1_ps(0.5f);

    /* Predict positions */
    __m256 pred_x = fmadd_ps(ax, _mm256_mul_ps(half, t2),
                              fmadd_ps(vx, t, x));
    __m256 pred_y = fmadd_ps(ay, _mm256_mul_ps(half, t2),
                              fmadd_ps(vy, t, y));

    /* Calculate danger based on speed and health */
    __m256 speed_sq = fmadd_ps(vy, vy, _mm256_mul_ps(vx, vx));
    __m256 speed = _mm256_sqrt_ps(speed_sq);

    __m256 health = _mm256_set_ps(
        threats[7]->health / threats[7]->max_health,
        threats[6]->health / threats[6]->max_health,
        threats[5]->health / threats[5]->max_health,
        threats[4]->health / threats[4]->max_health,
        threats[3]->health / threats[3]->max_health,
        threats[2]->health / threats[2]->max_health,
        threats[1]->health / threats[1]->max_health,
        threats[0]->health / threats[0]->max_health
    );

    /* Danger = speed * health * (in_combat_state ? 2 : 1) */
    __m256 danger = _mm256_mul_ps(speed, health);

    /* Store results */
    float __attribute__((aligned(32))) px[8], py[8], dng[8];
    _mm256_storeu_ps(px, pred_x);
    _mm256_storeu_ps(py, pred_y);
    _mm256_storeu_ps(dng, danger);

    for (int i = 0; i < 8; i++) {
        predictions[i].predicted_x = px[i];
        predictions[i].predicted_y = py[i];
        predictions[i].time_to_arrive = forecast_time;
        predictions[i].danger_level = dng[i];
        predictions[i].pattern = histories[i] ?
            pred_avx_analyze_pattern(histories[i]) : PATTERN_STATIONARY;
    }
}

/* ==========================================================================
 * COVER POSITION EVALUATION
 * ========================================================================== */

/* Evaluate 8 cover positions against a predicted threat */
static inline void pred_avx_evaluate_cover_8(
    __m256 cover_x, __m256 cover_y,
    float threat_predicted_x, float threat_predicted_y,
    float threat_vx, float threat_vy,
    float my_x, float my_y,
    __m256* cover_score
) {
    __m256 tx = _mm256_set1_ps(threat_predicted_x);
    __m256 ty = _mm256_set1_ps(threat_predicted_y);
    __m256 tvx = _mm256_set1_ps(threat_vx);
    __m256 tvy = _mm256_set1_ps(threat_vy);
    __m256 mx = _mm256_set1_ps(my_x);
    __m256 my_pos = _mm256_set1_ps(my_y);

    /* Distance from threat to cover */
    __m256 dx = _mm256_sub_ps(cover_x, tx);
    __m256 dy = _mm256_sub_ps(cover_y, ty);
    __m256 threat_dist = _mm256_sqrt_ps(fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx)));

    /* Distance from me to cover */
    __m256 mdx = _mm256_sub_ps(cover_x, mx);
    __m256 mdy = _mm256_sub_ps(cover_y, my_pos);
    __m256 my_dist = _mm256_sqrt_ps(fmadd_ps(mdy, mdy, _mm256_mul_ps(mdx, mdx)));

    /* Dot product: is threat moving toward this cover? */
    __m256 inv_dist = _mm256_div_ps(_mm256_set1_ps(1.0f),
        _mm256_add_ps(threat_dist, _mm256_set1_ps(0.001f)));
    __m256 norm_dx = _mm256_mul_ps(dx, inv_dist);
    __m256 norm_dy = _mm256_mul_ps(dy, inv_dist);

    __m256 threat_dir_dot = fmadd_ps(norm_dy, tvy, _mm256_mul_ps(norm_dx, tvx));

    /* Score: prefer cover that is:
     * - Far from threat (more time to reach)
     * - Close to me (can get there fast)
     * - Not in threat's movement direction
     */
    __m256 dist_score = _mm256_mul_ps(threat_dist, _mm256_set1_ps(0.1f));  /* Farther = better */
    __m256 access_score = _mm256_div_ps(_mm256_set1_ps(10.0f),
        _mm256_add_ps(my_dist, _mm256_set1_ps(1.0f)));  /* Closer = better */
    __m256 dir_score = _mm256_sub_ps(_mm256_set1_ps(1.0f), threat_dir_dot);  /* Away from threat path = better */

    *cover_score = _mm256_add_ps(_mm256_add_ps(dist_score, access_score), dir_score);
}

/* ==========================================================================
 * PROJECTILE AVOIDANCE PREDICTION
 * ========================================================================== */

/* Check if entity is in danger from 8 bullets */
static inline __m256 pred_avx_bullet_danger_8(
    float entity_x, float entity_y,
    __m256 bullet_x, __m256 bullet_y,
    __m256 bullet_vx, __m256 bullet_vy,
    float danger_radius,
    float lookahead_time
) {
    __m256 ex = _mm256_set1_ps(entity_x);
    __m256 ey = _mm256_set1_ps(entity_y);
    __m256 t = _mm256_set1_ps(lookahead_time);
    __m256 r2 = _mm256_set1_ps(danger_radius * danger_radius);

    /* Predict bullet positions */
    __m256 pred_bx = fmadd_ps(bullet_vx, t, bullet_x);
    __m256 pred_by = fmadd_ps(bullet_vy, t, bullet_y);

    /* Check if predicted position is within danger radius */
    __m256 dx = _mm256_sub_ps(pred_bx, ex);
    __m256 dy = _mm256_sub_ps(pred_by, ey);
    __m256 dist_sq = fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx));

    return _mm256_cmp_ps(dist_sq, r2, _CMP_LT_OQ);
}

/* Calculate best dodge direction from 8 incoming bullets */
static inline void pred_avx_dodge_direction_8(
    float entity_x, float entity_y,
    __m256 bullet_x, __m256 bullet_y,
    __m256 bullet_vx, __m256 bullet_vy,
    __m256 bullet_active,      /* Mask of active bullets */
    float* dodge_x, float* dodge_y
) {
    __m256 ex = _mm256_set1_ps(entity_x);
    __m256 ey = _mm256_set1_ps(entity_y);

    /* Direction from bullet to entity (perpendicular to escape) */
    __m256 dx = _mm256_sub_ps(ex, bullet_x);
    __m256 dy = _mm256_sub_ps(ey, bullet_y);

    /* Perpendicular to bullet velocity (dodge direction) */
    /* Perpendicular of (vx, vy) is (-vy, vx) */
    __m256 perp_x = _mm256_sub_ps(_mm256_setzero_ps(), bullet_vy);
    __m256 perp_y = bullet_vx;

    /* Choose perpendicular direction that moves entity further from bullet path */
    /* Dot product of (entity - bullet) with perp */
    __m256 dot = fmadd_ps(dy, perp_y, _mm256_mul_ps(dx, perp_x));

    /* Flip perpendicular if pointing wrong way */
    __m256 flip_mask = _mm256_cmp_ps(dot, _mm256_setzero_ps(), _CMP_LT_OQ);
    perp_x = _mm256_blendv_ps(perp_x, _mm256_sub_ps(_mm256_setzero_ps(), perp_x), flip_mask);
    perp_y = _mm256_blendv_ps(perp_y, _mm256_sub_ps(_mm256_setzero_ps(), perp_y), flip_mask);

    /* Weight by proximity to bullet (closer = more urgent) */
    __m256 dist_sq = fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx));
    __m256 weight = _mm256_div_ps(_mm256_set1_ps(100.0f),
        _mm256_add_ps(dist_sq, _mm256_set1_ps(1.0f)));

    /* Apply active mask */
    weight = _mm256_and_ps(weight, bullet_active);

    /* Sum weighted dodge directions */
    __m256 weighted_x = _mm256_mul_ps(perp_x, weight);
    __m256 weighted_y = _mm256_mul_ps(perp_y, weight);

    float sum_x = hsum256_ps(weighted_x);
    float sum_y = hsum256_ps(weighted_y);

    /* Normalize */
    float len = sqrtf(sum_x * sum_x + sum_y * sum_y);
    if (len > 0.001f) {
        *dodge_x = sum_x / len;
        *dodge_y = sum_y / len;
    } else {
        *dodge_x = 0;
        *dodge_y = 0;
    }
}

/* ==========================================================================
 * PREDICTION API
 * ========================================================================== */

/* Update movement history for an entity */
void pred_update_history(MovementHistory* hist, const Entity* entity);

/* Predict where target will be in t frames */
void pred_forecast_position(
    const Entity* target,
    const MovementHistory* hist,
    float frames_ahead,
    float* out_x, float* out_y,
    float* out_confidence
);

/* Calculate aim point for projectile intercept */
void pred_calculate_aim(
    const Entity* shooter,
    const Entity* target,
    const MovementHistory* target_hist,
    float bullet_speed,
    float* aim_x, float* aim_y
);

/* Evaluate danger level from incoming threats */
float pred_evaluate_danger(
    const Entity* self,
    const GameState* game,
    float lookahead_frames
);

/* Find safest direction to move */
void pred_find_safe_direction(
    const Entity* self,
    const GameState* game,
    float* safe_x, float* safe_y
);

#endif /* SHOOTER_PREDICTION_H */
