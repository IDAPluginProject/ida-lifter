/*
 * Advanced Enemy AI System Header
 * Behavior trees, neural networks, GOAP, and sophisticated tactical AI.
 */

#ifndef SHOOTER_ENEMY_AI_ADVANCED_H
#define SHOOTER_ENEMY_AI_ADVANCED_H

#include "../types.h"
#include "../config.h"

/*
 * Initialize the advanced enemy AI system.
 * Must be called once before using update functions.
 */
void init_enemy_ai_advanced(GameState* game);

/*
 * Update a single enemy using the advanced AI system.
 * Uses behavior trees, neural networks, and AVX batch processing.
 */
void update_enemy_ai_advanced(GameState* game, int entity_id);

/*
 * Batch update all enemies using AVX operations.
 * More efficient than individual updates for large entity counts.
 */
void update_enemies_batch_avx(GameState* game);

/*
 * Shutdown and cleanup AI resources.
 */
void shutdown_enemy_ai_advanced(void);

#endif /* SHOOTER_ENEMY_AI_ADVANCED_H */
