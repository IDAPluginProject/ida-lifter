CC = clang-20
CFLAGS = -target x86_64-apple-darwin -O3 -ffast-math -Wall -Wextra -mavx2 -mfma -msse4.1 -std=c11 -D_POSIX_C_SOURCE=200112L
LDFLAGS = -target x86_64-apple-darwin -lm

SRC_DIR = src
OBJ_DIR = obj

# Main binaries
BIN_REF = decompiler_ref
BIN_SHOOTER = shooter

# Source files (excluding mains)
COMMON_SRCS = $(SRC_DIR)/vis.c $(SRC_DIR)/bullet.c $(SRC_DIR)/fluid.c $(SRC_DIR)/asteroid.c $(SRC_DIR)/interplanetary.c
COMMON_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(COMMON_SRCS))

# Main-specific
REF_MAIN = $(OBJ_DIR)/main.o
SHOOTER_MAIN = $(OBJ_DIR)/shooter_main.o

all: $(BIN_REF) $(BIN_SHOOTER)

$(BIN_REF): $(COMMON_OBJS) $(REF_MAIN)
	$(CC) $^ -o $@ $(LDFLAGS)

$(BIN_SHOOTER): $(COMMON_OBJS) $(SHOOTER_MAIN)
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_REF) $(BIN_SHOOTER)

.PHONY: all clean
