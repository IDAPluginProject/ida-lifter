#include "common.h"
#include "math_avx.h"
#include <string.h>

#define NUM_PARTICLES 128
#define ATTRACTION_STRENGTH 0.3f
#define REPULSION_STRENGTH 2.0f
#define MAX_FORCE 5.0f
#define DAMPING 0.98f

// Particle swarm dynamics using AVX
void update_particles_avx(__m256* px, __m256* py, __m256* vx, __m256* vy,
                         float cx, float cy, int count) {
    __m256 center_x = _mm256_set1_ps(cx);
    __m256 center_y = _mm256_set1_ps(cy);
    __m256 attraction = _mm256_set1_ps(ATTRACTION_STRENGTH);
    __m256 repulsion = _mm256_set1_ps(REPULSION_STRENGTH);
    __m256 max_force_vec = _mm256_set1_ps(MAX_FORCE);
    __m256 damping_vec = _mm256_set1_ps(DAMPING);
    __m256 dt = _mm256_set1_ps(0.016f);
    __m256 softening = _mm256_set1_ps(0.5f);

    for (int i = 0; i < count; i++) {
        __m256 fx = _mm256_setzero_ps();
        __m256 fy = _mm256_setzero_ps();

        // Attraction to center
        __m256 dx_center = _mm256_sub_ps(center_x, px[i]);
        __m256 dy_center = _mm256_sub_ps(center_y, py[i]);
        __m256 dist_center_sq = _mm256_fmadd_ps(dx_center, dx_center,
                                                _mm256_mul_ps(dy_center, dy_center));
        __m256 dist_center = _mm256_sqrt_ps(dist_center_sq);
        __m256 force_center = _mm256_mul_ps(attraction, dist_center);

        __m256 dir_x = _mm256_div_ps(dx_center, _mm256_add_ps(dist_center, softening));
        __m256 dir_y = _mm256_div_ps(dy_center, _mm256_add_ps(dist_center, softening));

        fx = _mm256_fmadd_ps(force_center, dir_x, fx);
        fy = _mm256_fmadd_ps(force_center, dir_y, fy);

        // Repulsion from other particles
        for (int j = 0; j < count; j++) {
            if (i == j) continue;

            __m256 dx = _mm256_sub_ps(px[i], px[j]);
            __m256 dy = _mm256_sub_ps(py[i], py[j]);
            __m256 dist_sq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
            __m256 dist = _mm256_sqrt_ps(_mm256_add_ps(dist_sq, softening));
            __m256 inv_dist = _mm256_rcp_ps(dist);
            __m256 force_rep = _mm256_mul_ps(repulsion, _mm256_mul_ps(inv_dist, inv_dist));

            __m256 rep_dir_x = _mm256_mul_ps(dx, inv_dist);
            __m256 rep_dir_y = _mm256_mul_ps(dy, inv_dist);

            fx = _mm256_fmadd_ps(force_rep, rep_dir_x, fx);
            fy = _mm256_fmadd_ps(force_rep, rep_dir_y, fy);
        }

        // Clamp forces
        __m256 force_mag = _mm256_sqrt_ps(_mm256_fmadd_ps(fx, fx, _mm256_mul_ps(fy, fy)));
        __m256 clamped = _mm256_min_ps(force_mag, max_force_vec);
        __m256 scale = _mm256_div_ps(clamped, _mm256_add_ps(force_mag, softening));
        fx = _mm256_mul_ps(fx, scale);
        fy = _mm256_mul_ps(fy, scale);

        // Update velocities and positions
        vx[i] = _mm256_fmadd_ps(fx, dt, vx[i]);
        vy[i] = _mm256_fmadd_ps(fy, dt, vy[i]);
        vx[i] = _mm256_mul_ps(vx[i], damping_vec);
        vy[i] = _mm256_mul_ps(vy[i], damping_vec);

        px[i] = _mm256_fmadd_ps(vx[i], dt, px[i]);
        py[i] = _mm256_fmadd_ps(vy[i], dt, py[i]);
    }
}

void run_particle_sim() {
    Viewport* vp = create_viewport();

    int num_groups = NUM_PARTICLES / 8;
    __m256* px = (__m256*)aligned_alloc(32, num_groups * sizeof(__m256));
    __m256* py = (__m256*)aligned_alloc(32, num_groups * sizeof(__m256));
    __m256* vx = (__m256*)aligned_alloc(32, num_groups * sizeof(__m256));
    __m256* vy = (__m256*)aligned_alloc(32, num_groups * sizeof(__m256));

    float cx = vp->width / 2.0f;
    float cy = vp->height / 2.0f;

    // Initialize particles in a ring
    float temp_x[NUM_PARTICLES], temp_y[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float angle = 2.0f * PI * i / NUM_PARTICLES;
        temp_x[i] = cx + cosf(angle) * 20.0f;
        temp_y[i] = cy + sinf(angle) * 10.0f;
    }

    for (int i = 0; i < num_groups; i++) {
        px[i] = _mm256_loadu_ps(&temp_x[i * 8]);
        py[i] = _mm256_loadu_ps(&temp_y[i * 8]);
        vx[i] = _mm256_setzero_ps();
        vy[i] = _mm256_setzero_ps();
    }

    int frames = SIM_DURATION_SEC * FPS;
    const char* particle_char = "•";

    for (int f = 0; f < frames; f++) {
        clear_buffer(vp);

        // Title
        draw_box(vp, 0, 0, vp->width - 1, vp->height - 1, "single");
        const char* title = "⚛ PARTICLE SWARM DYNAMICS (AVX) ⚛";
        draw_string(vp, (vp->width - strlen(title)) / 2, 1, title);

        // Pulsating attractor at center
        float pulse = 1.0f + 0.3f * sinf(f * 0.1f);
        int attractor_size = (int)(2.0f * pulse);
        draw_filled_circle(vp, (int)cx, (int)cy, attractor_size, "▓");
        draw_pixel(vp, (int)cx, (int)cy, "◎");

        // Update physics
        update_particles_avx(px, py, vx, vy, cx, cy, num_groups);

        // Render particles
        for (int i = 0; i < num_groups; i++) {
            _mm256_storeu_ps(&temp_x[i * 8], px[i]);
            _mm256_storeu_ps(&temp_y[i * 8], py[i]);
        }

        for (int i = 0; i < NUM_PARTICLES; i++) {
            int x = (int)temp_x[i];
            int y = (int)temp_y[i];

            if (x >= 1 && x < vp->width - 1 && y >= 2 && y < vp->height - 2) {
                // Draw connection to center occasionally
                if (i % 8 == f % 8) {
                    draw_line(vp, x, y, (int)cx, (int)cy, "·");
                }
                draw_pixel(vp, x, y, particle_char);
            }
        }

        // Stats
        char stats[128];
        snprintf(stats, 128, "Particles: %d │ Attraction: %.2f │ Repulsion: %.2f │ Frame: %d",
                 NUM_PARTICLES, ATTRACTION_STRENGTH, REPULSION_STRENGTH, f + 1);
        draw_string(vp, 3, vp->height - 2, stats);

        render_buffer(vp);
        sleep_ms(1000 / FPS);
    }

    free(px);
    free(py);
    free(vx);
    free(vy);
    free_viewport(vp);
}
