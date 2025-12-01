#include <stdio.h>
#include "common.h"
#include "math_avx.h"
#include <string.h>

#define MAX_AST_TRAIL 1000

typedef struct {
    float x, y;
    int age;
} AsteroidTrail;

void update_asteroid_sse(
    Vec4* asteroid_pos, Vec4* asteroid_vel,
    const Star* star, const Planet planets[3],
    float dt, float* out_time_dilation
) {
    __m128 a_x = _mm_set1_ps(asteroid_pos->x);
    __m128 a_y = _mm_set1_ps(asteroid_pos->y);
    __m128 a_z = _mm_set1_ps(asteroid_pos->z);

    __m128 b_x = _mm_set_ps(planets[2].pos.x, planets[1].pos.x, planets[0].pos.x, star->pos.x);
    __m128 b_y = _mm_set_ps(planets[2].pos.y, planets[1].pos.y, planets[0].pos.y, star->pos.y);
    __m128 b_z = _mm_set_ps(planets[2].pos.z, planets[1].pos.z, planets[0].pos.z, star->pos.z);

    __m128 masses = _mm_set_ps(planets[2].mass, planets[1].mass, planets[0].mass, star->mass);

    __m128 rx = _mm_sub_ps(b_x, a_x);
    __m128 ry = _mm_sub_ps(b_y, a_y);
    __m128 rz = _mm_sub_ps(b_z, a_z);

    __m128 r2 = _mm_mul_ps(rx, rx);
    r2 = fmadd_sse(ry, ry, r2);
    r2 = fmadd_sse(rz, rz, r2);

    __m128 r_inv = _mm_rsqrt_ps(r2);
    __m128 half = _mm_set1_ps(0.5f);
    __m128 three_half = _mm_set1_ps(1.5f);
    __m128 r_inv2 = _mm_mul_ps(r_inv, r_inv);
    r_inv = _mm_mul_ps(r_inv, _mm_sub_ps(three_half, _mm_mul_ps(half, _mm_mul_ps(r2, r_inv2))));

    __m128 r_inv3 = _mm_mul_ps(r_inv, _mm_mul_ps(r_inv, r_inv));

    __m128 G = _mm_set1_ps(G_CONST);
    __m128 scalar = _mm_mul_ps(G, _mm_mul_ps(masses, r_inv3));

    __m128 ax = _mm_mul_ps(rx, scalar);
    __m128 ay = _mm_mul_ps(ry, scalar);
    __m128 az = _mm_mul_ps(rz, scalar);

    float acc_x = hsum128_ps(ax);
    float acc_y = hsum128_ps(ay);
    float acc_z = hsum128_ps(az);

    __m128 v_dt = _mm_set1_ps(dt);
    __m128 acc_vec = _mm_set_ps(0, acc_z, acc_y, acc_x);
    __m128 vel = _mm_loadu_ps(&asteroid_vel->x);
    __m128 pos = _mm_loadu_ps(&asteroid_pos->x);

    vel = fmadd_sse(acc_vec, v_dt, vel);
    pos = fmadd_sse(vel, v_dt, pos);

    _mm_storeu_ps(&asteroid_vel->x, vel);
    _mm_storeu_ps(&asteroid_pos->x, pos);

    __m128 r2_star_v = _mm_set1_ps(_mm_cvtss_f32(r2));
    __m128 r_star_v = sqrt_nr_sse(r2_star_v);

    __m128 v_2gm = _mm_set1_ps(2.0f * G_CONST * star->mass);
    __m128 v_c2 = _mm_set1_ps(C_LIGHT * C_LIGHT);
    __m128 v_rs = _mm_div_ps(v_2gm, v_c2);
    __m128 ratio = _mm_div_ps(v_rs, r_star_v);
    __m128 one_minus = _mm_sub_ps(_mm_set1_ps(1.0f), ratio);
    __m128 dilation_factor = sqrt_nr_sse(one_minus);

    *out_time_dilation = dt * _mm_cvtss_f32(dilation_factor);
}

void run_asteroid_sim() {
    Viewport* vp = create_viewport();
    
    Star star = {1.989e30f, 696340000.0f, {0,0,0,0}};
    Planet p1 = {5.972e24f, 6371000.0f, 1.496e11f, 1.99e-7f, 0, {1.496e11f, 0, 0, 0}};
    Planet p2 = {6.39e23f, 3389500.0f, 2.279e11f, 1.05e-7f, 0, {-2.279e11f, 0, 0, 0}};
    Planet planets[3] = {p1, p2, p1};

    Vec4 ast_pos = {-2.0e11f, 1.0e11f, 0, 0};
    Vec4 ast_vel = {40000.0f, -10000.0f, 0, 0};

    float dt = 20000.0f;
    float dilation = 0;

    int cx = vp->width / 2;
    int cy = vp->height / 2;
    float scale = (vp->width < vp->height ? vp->width : vp->height) * 0.00000000018f;

    AsteroidTrail trail[MAX_AST_TRAIL];
    int trail_idx = 0;
    for (int i = 0; i < MAX_AST_TRAIL; i++) {
        trail[i].age = 0;
    }

    int frames = SIM_DURATION_SEC * FPS;
    for (int f = 0; f < frames; f++) {
        clear_buffer(vp);
        draw_box(vp, 0, 0, vp->width - 1, vp->height - 1, "single");
        
        char title[128];
        snprintf(title, 128, "━━━ ☄ RELATIVISTIC ASTEROID TRAJECTORY ☄ ━━━");
        draw_string(vp, (vp->width - strlen(title)) / 2, 1, title);

        update_asteroid_sse(&ast_pos, &ast_vel, &star, planets, dt, &dilation);

        // Age trail
        for (int i = 0; i < MAX_AST_TRAIL; i++) {
            if (trail[i].age > 0) trail[i].age--;
        }

        // Add to trail
        int ax = cx + (int)(ast_pos.x * scale);
        int ay = cy + (int)(ast_pos.y * scale * 0.5f);
        trail[trail_idx].x = ax;
        trail[trail_idx].y = ay;
        trail[trail_idx].age = 150;
        trail_idx = (trail_idx + 1) % MAX_AST_TRAIL;

        // Draw trail with gradient
        for (int i = 0; i < MAX_AST_TRAIL; i++) {
            if (trail[i].age > 0) {
                const char* fade[] = {"━", "─", "╌", "┄", "·", " "};
                int idx = (trail[i].age > 120) ? 0 : (trail[i].age > 90) ? 1 : 
                         (trail[i].age > 60) ? 2 : (trail[i].age > 30) ? 3 : 
                         (trail[i].age > 10) ? 4 : 5;
                draw_pixel(vp, trail[i].x, trail[i].y, fade[idx]);
            }
        }

        // Draw Star
        draw_filled_circle(vp, cx, cy, 2, "▒");
        draw_pixel(vp, cx, cy, "⊛");

        // Draw Planets with orbits
        int px1 = cx + (int)(p1.pos.x * scale);
        int py1 = cy + (int)(p1.pos.y * scale * 0.5f);
        draw_circle(vp, cx, cy, abs(px1 - cx), "∘");
        draw_filled_circle(vp, px1, py1, 1, "▓");
        draw_pixel(vp, px1, py1, "◉");

        int px2 = cx + (int)(p2.pos.x * scale);
        int py2 = cy + (int)(p2.pos.y * scale * 0.5f);
        draw_circle(vp, cx, cy, abs(px2 - cx), "∘");
        draw_pixel(vp, px2, py2, "○");

        // Draw Asteroid with burst effect
        const char* ast_glyph = (f % 8 < 4) ? "☄" : "✦";
        draw_pixel(vp, ax, ay, ast_glyph);
        if (f % 4 == 0) {
            draw_pixel(vp, ax-1, ay, "∗");
            draw_pixel(vp, ax+1, ay, "∗");
        }

        // Velocity vector
        int vx = ax + (int)(ast_vel.x * 0.0001f);
        int vy = ay + (int)(ast_vel.y * 0.0001f * 0.5f);
        draw_line(vp, ax, ay, vx, vy, "→");

        // Stats
        float speed = sqrtf(ast_vel.x*ast_vel.x + ast_vel.y*ast_vel.y + ast_vel.z*ast_vel.z);
        char stats[256];
        snprintf(stats, 256, "Position: (%.2e, %.2e) km | Velocity: %.2e m/s | Time Dilation: %.10f",
                 ast_pos.x/1000.0f, ast_pos.y/1000.0f, speed, dilation/dt);
        draw_string(vp, 3, vp->height - 2, stats);

        render_buffer(vp);
        sleep_ms(1000 / FPS);
    }
    
    free_viewport(vp);
}
