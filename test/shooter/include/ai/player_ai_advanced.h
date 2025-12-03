/*
 * Advanced Player AI System Header
 * Sophisticated tactical planning, threat assessment, and AVX optimization.
 */

#ifndef SHOOTER_PLAYER_AI_ADVANCED_H
#define SHOOTER_PLAYER_AI_ADVANCED_H

#include "../types.h"
#include "../config.h"

/*
 * Initialize the advanced player AI system.
 * Must be called once before using update functions.
 */
void init_player_ai_advanced(GameState* game);

/*
 * Update the player using the advanced AI system.
 * Uses AVX batch threat assessment and tactical planning.
 */
void update_player_ai_advanced(GameState* game);

/*
 * Shutdown and cleanup player AI resources.
 */
void shutdown_player_ai_advanced(void);

#endif /* SHOOTER_PLAYER_AI_ADVANCED_H */
