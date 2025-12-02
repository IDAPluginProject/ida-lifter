#include "common.h"
#include "math_avx.h"
#include <string.h>

#define NUM_BODIES 16
#define GRAV_CONST 0.5f
#define DAMPING 0.995f
#define SOFTENING 0.1f

typedef struct {
    __m256 x, y;      // Positions (8 bodies per vector)
    __m256 vx, vy;    // Velocities
    __m256 mass;      // Masses
} BodyGroup;

// Calculate gravitational forces using AVX2
void compute_forces_avx(BodyGroup* bodies, __m256* fx, __m256* fy) {
    __m256 zero = _mm256_setzero_ps();
    __m256 softening_sq = _mm256_set1_ps(SOFTENING * SOFTENING);

    // Initialize force accumulator
    *fx = zero;
    *fy = zero;

    // Compute forces between all body pairs
    for (int j = 0; j < 2; j++) {
        BodyGroup* other = &bodies[j];

        // Distance vectors
        __m256 dx = _mm256_sub_ps(other->x, bodies[0].x);
        __m256 dy = _mm256_sub_ps(other->y, bodies[0].y);

        // Distance squared with softening
        __m256 dist_sq = _mm256_mul_ps(dx, dx);
        dist_sq = _mm256_fmadd_ps(dy, dy, dist_sq);
        dist_sq = _mm256_add_ps(dist_sq, softening_sq);

        // dist^(-3/2) using rsqrt approximation
        __m256 inv_dist = _mm256_rsqrt_ps(dist_sq);
        __m256 inv_dist3 = _mm256_mul_ps(_mm256_mul_ps(inv_dist, inv_dist), inv_dist);

        // Force magnitude: G * m1 * m2 / r^3
        __m256 force_mag = _mm256_mul_ps(other->mass, inv_dist3);
        force_mag = _mm256_mul_ps(force_mag, _mm256_set1_ps(GRAV_CONST));

        // Accumulate force components
        *fx = _mm256_fmadd_ps(dx, force_mag, *fx);
        *fy = _mm256_fmadd_ps(dy, force_mag, *fy);
    }
}

// Update positions using AVX velocity integration
void integrate_avx(BodyGroup* bodies, float dt) {
    __m256 fx, fy;
    compute_forces_avx(bodies, &fx, &fy);

    __m256 dt_vec = _mm256_set1_ps(dt);
    __m256 damping_vec = _mm256_set1_ps(DAMPING);

    // Update velocities: v += (F / m) * dt
    __m256 inv_mass = _mm256_rcp_ps(bodies[0].mass);
    __m256 ax = _mm256_mul_ps(fx, inv_mass);
    __m256 ay = _mm256_mul_ps(fy, inv_mass);

    bodies[0].vx = _mm256_fmadd_ps(ax, dt_vec, bodies[0].vx);
    bodies[0].vy = _mm256_fmadd_ps(ay, dt_vec, bodies[0].vy);

    // Apply damping
    bodies[0].vx = _mm256_mul_ps(bodies[0].vx, damping_vec);
    bodies[0].vy = _mm256_mul_ps(bodies[0].vy, damping_vec);

    // Update positions: p += v * dt
    bodies[0].x = _mm256_fmadd_ps(bodies[0].vx, dt_vec, bodies[0].x);
    bodies[0].y = _mm256_fmadd_ps(bodies[0].vy, dt_vec, bodies[0].y);

    // Repeat for second group
    compute_forces_avx(&bodies[1], &fx, &fy);
    inv_mass = _mm256_rcp_ps(bodies[1].mass);
    ax = _mm256_mul_ps(fx, inv_mass);
    ay = _mm256_mul_ps(fy, inv_mass);

    bodies[1].vx = _mm256_fmadd_ps(ax, dt_vec, bodies[1].vx);
    bodies[1].vy = _mm256_fmadd_ps(ay, dt_vec, bodies[1].vy);
    bodies[1].vx = _mm256_mul_ps(bodies[1].vx, damping_vec);
    bodies[1].vy = _mm256_mul_ps(bodies[1].vy, damping_vec);

    bodies[1].x = _mm256_fmadd_ps(bodies[1].vx, dt_vec, bodies[1].x);
    bodies[1].y = _mm256_fmadd_ps(bodies[1].vy, dt_vec, bodies[1].y);
}

void run_nbody_sim() {
    Viewport* vp = create_viewport();

    BodyGroup bodies[2];
    float cx = vp->width / 2.0f;
    float cy = vp->height / 2.0f;

    // Initialize 16 bodies in two groups of 8
    float pos_x[16], pos_y[16], vel_x[16], vel_y[16], masses[16];

    for (int i = 0; i < NUM_BODIES; i++) {
        float angle = 2.0f * PI * i / NUM_BODIES;
        float radius = 15.0f + (i % 4) * 3.0f;
        pos_x[i] = cx + cosf(angle) * radius;
        pos_y[i] = cy + sinf(angle) * radius * 0.5f;
        vel_x[i] = -sinf(angle) * 0.5f;
        vel_y[i] = cosf(angle) * 0.25f;
        masses[i] = 0.5f + (i % 3) * 0.5f;
    }

    // Load into AVX registers
    bodies[0].x = _mm256_loadu_ps(pos_x);
    bodies[0].y = _mm256_loadu_ps(pos_y);
    bodies[0].vx = _mm256_loadu_ps(vel_x);
    bodies[0].vy = _mm256_loadu_ps(vel_y);
    bodies[0].mass = _mm256_loadu_ps(masses);

    bodies[1].x = _mm256_loadu_ps(pos_x + 8);
    bodies[1].y = _mm256_loadu_ps(pos_y + 8);
    bodies[1].vx = _mm256_loadu_ps(vel_x + 8);
    bodies[1].vy = _mm256_loadu_ps(vel_y + 8);
    bodies[1].mass = _mm256_loadu_ps(masses + 8);

    const char* body_chars[] = {"●", "○", "◉", "◎"};
    int frames = SIM_DURATION_SEC * FPS;

    for (int f = 0; f < frames; f++) {
        clear_buffer(vp);

        // Title
        draw_box(vp, 0, 0, vp->width - 1, vp->height - 1, "double");
        const char* title = "║ N-BODY GRAVITATIONAL SIMULATION (AVX2) ║";
        draw_string(vp, (vp->width - strlen(title)) / 2, 1, title);

        // Integrate physics
        integrate_avx(bodies, 0.016f);

        // Store back to arrays for rendering
        _mm256_storeu_ps(pos_x, bodies[0].x);
        _mm256_storeu_ps(pos_y, bodies[0].y);
        _mm256_storeu_ps(pos_x + 8, bodies[1].x);
        _mm256_storeu_ps(pos_y + 8, bodies[1].y);

        // Draw center of mass indicator
        draw_filled_circle(vp, (int)cx, (int)cy, 2, "░");
        draw_pixel(vp, (int)cx, (int)cy, "+");

        // Draw bodies with trails
        for (int i = 0; i < NUM_BODIES; i++) {
            int x = (int)pos_x[i];
            int y = (int)pos_y[i];

            // Draw trail
            if (f > 0) {
                draw_filled_circle(vp, x, y, 1, "·");
            }

            // Draw body
            const char* ch = body_chars[i % 4];
            draw_pixel(vp, x, y, ch);
        }

        // Stats
        char stats[128];
        snprintf(stats, 128, "Bodies: %d │ Frame: %d/%d │ G=%.2f",
                 NUM_BODIES, f + 1, frames, GRAV_CONST);
        draw_string(vp, 3, vp->height - 2, stats);

        render_buffer(vp);
        sleep_ms(1000 / FPS);
    }

    free_viewport(vp);
}
