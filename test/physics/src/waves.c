#include "common.h"
#include "math_avx.h"
#include <string.h>

#define WAVE_W 80
#define WAVE_H 40
#define WAVE_SIZE (WAVE_W * WAVE_H)

// 2D wave propagation using AVX2
void propagate_wave_avx(float* current, float* prev, float* next, float c, float dt, float dx) {
    __m256 c_sq = _mm256_set1_ps(c * c);
    __m256 dt_sq = _mm256_set1_ps(dt * dt);
    __m256 dx_sq = _mm256_set1_ps(dx * dx);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 coeff = _mm256_div_ps(_mm256_mul_ps(c_sq, dt_sq), dx_sq);
    __m256 damping = _mm256_set1_ps(0.999f);

    for (int y = 1; y < WAVE_H - 1; y++) {
        for (int x = 1; x < WAVE_W - 8; x += 8) {
            int idx = y * WAVE_W + x;

            // Load current values
            __m256 curr = _mm256_loadu_ps(&current[idx]);
            __m256 old = _mm256_loadu_ps(&prev[idx]);

            // Load neighbors for Laplacian
            __m256 left = _mm256_loadu_ps(&current[idx - 1]);
            __m256 right = _mm256_loadu_ps(&current[idx + 1]);
            __m256 up = _mm256_loadu_ps(&current[idx - WAVE_W]);
            __m256 down = _mm256_loadu_ps(&current[idx + WAVE_W]);

            // Compute Laplacian: ∇²u = (left + right + up + down - 4*current)
            __m256 laplacian = _mm256_add_ps(left, right);
            laplacian = _mm256_add_ps(laplacian, up);
            laplacian = _mm256_add_ps(laplacian, down);
            __m256 center_term = _mm256_mul_ps(curr, _mm256_set1_ps(4.0f));
            laplacian = _mm256_sub_ps(laplacian, center_term);

            // Wave equation: u_new = 2*u - u_old + c²*dt²/dx² * ∇²u
            __m256 result = _mm256_mul_ps(two, curr);
            result = _mm256_sub_ps(result, old);
            result = _mm256_fmadd_ps(coeff, laplacian, result);

            // Apply damping
            result = _mm256_mul_ps(result, damping);

            _mm256_storeu_ps(&next[idx], result);
        }
    }
}

void run_wave_sim() {
    Viewport* vp = create_viewport();

    float* wave_curr = (float*)aligned_alloc(32, WAVE_SIZE * sizeof(float));
    float* wave_prev = (float*)aligned_alloc(32, WAVE_SIZE * sizeof(float));
    float* wave_next = (float*)aligned_alloc(32, WAVE_SIZE * sizeof(float));

    memset(wave_curr, 0, WAVE_SIZE * sizeof(float));
    memset(wave_prev, 0, WAVE_SIZE * sizeof(float));
    memset(wave_next, 0, WAVE_SIZE * sizeof(float));

    float c = 1.5f;  // Wave speed
    float dt = 0.05f;
    float dx = 1.0f;

    int frames = SIM_DURATION_SEC * FPS;

    for (int f = 0; f < frames; f++) {
        clear_buffer(vp);

        // Title
        draw_box(vp, 0, 0, vp->width - 1, vp->height - 1, "single");
        const char* title = "═══ WAVE INTERFERENCE PATTERN (AVX2) ═══";
        draw_string(vp, (vp->width - strlen(title)) / 2, 1, title);

        // Add wave sources (moving)
        int s1_x = WAVE_W / 2 + (int)(cosf(f * 0.1f) * 15);
        int s1_y = WAVE_H / 2 + (int)(sinf(f * 0.1f) * 7);
        int s2_x = WAVE_W / 2 + (int)(cosf(f * 0.15f + PI) * 15);
        int s2_y = WAVE_H / 2 + (int)(sinf(f * 0.15f + PI) * 7);

        if (s1_x >= 0 && s1_x < WAVE_W && s1_y >= 0 && s1_y < WAVE_H) {
            wave_curr[s1_y * WAVE_W + s1_x] = sinf(f * 0.5f) * 5.0f;
        }
        if (s2_x >= 0 && s2_x < WAVE_W && s2_y >= 0 && s2_y < WAVE_H) {
            wave_curr[s2_y * WAVE_W + s2_x] = sinf(f * 0.7f) * 5.0f;
        }

        // Propagate wave using AVX2
        propagate_wave_avx(wave_curr, wave_prev, wave_next, c, dt, dx);

        // Swap buffers
        float* temp = wave_prev;
        wave_prev = wave_curr;
        wave_curr = wave_next;
        wave_next = temp;

        // Render wave field
        const char* intensity[] = {" ", "░", "▒", "▓", "█"};
        for (int y = 0; y < vp->height - 4; y++) {
            for (int x = 0; x < vp->width - 2; x++) {
                int wx = x * WAVE_W / (vp->width - 2);
                int wy = y * WAVE_H / (vp->height - 4);

                if (wx >= 0 && wx < WAVE_W && wy >= 0 && wy < WAVE_H) {
                    float val = wave_curr[wy * WAVE_W + wx];
                    int level = (int)((fabsf(val) + 1.0f) * 2.5f);
                    if (level < 0) level = 0;
                    if (level > 4) level = 4;

                    draw_pixel(vp, x + 1, y + 2, intensity[level]);
                }
            }
        }

        // Draw source indicators
        int screen_s1_x = s1_x * (vp->width - 2) / WAVE_W + 1;
        int screen_s1_y = s1_y * (vp->height - 4) / WAVE_H + 2;
        int screen_s2_x = s2_x * (vp->width - 2) / WAVE_W + 1;
        int screen_s2_y = s2_y * (vp->height - 4) / WAVE_H + 2;

        draw_pixel(vp, screen_s1_x, screen_s1_y, "⊕");
        draw_pixel(vp, screen_s2_x, screen_s2_y, "⊗");

        // Stats
        char stats[128];
        snprintf(stats, 128, "Wave Speed: %.2f │ Grid: %dx%d │ Frame: %d/%d",
                 c, WAVE_W, WAVE_H, f + 1, frames);
        draw_string(vp, 3, vp->height - 2, stats);

        render_buffer(vp);
        sleep_ms(1000 / FPS);
    }

    free(wave_curr);
    free(wave_prev);
    free(wave_next);
    free_viewport(vp);
}
