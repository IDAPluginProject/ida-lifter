#include "common.h"
#include <string.h>
#include <time.h>

int main() {
    printf("\033[2J\033[H"); // Clear screen and home cursor
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   AVX PHYSICS SIMULATION TESTBED - IDA Decompiler Test   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    printf("Running 4 SIMD-Intensive Physics Simulations...\n");
    printf("Each simulation: ~10 seconds\n\n");
    sleep_ms(2000);

    // 1. N-Body Gravitational Simulation (AVX-512)
    printf("▶ Simulation 1/4: N-Body Gravity (AVX-512)...\n");
    run_nbody_sim();
    sleep_ms(1000);

    // 2. Wave Interference Pattern (AVX2)
    printf("▶ Simulation 2/4: Wave Interference (AVX2)...\n");
    run_wave_sim();
    sleep_ms(1000);

    // 3. Particle Swarm Dynamics (AVX)
    printf("▶ Simulation 3/4: Particle Swarm (AVX)...\n");
    run_particle_sim();
    sleep_ms(1000);

    // 4. Fluid Vortex Dynamics (AVX2)
    printf("▶ Simulation 4/4: Fluid Vortex (AVX2)...\n");
    run_fluid_sim();
    sleep_ms(1000);

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║              All simulations completed!                  ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    return 0;
}
