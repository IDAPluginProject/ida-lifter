#include "common.h"
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <wchar.h>

// Get terminal size
static void get_terminal_size(int* width, int* height) {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
        *width = w.ws_col;
        *height = w.ws_row;
    } else {
        *width = 80;
        *height = 40;
    }
}

Viewport* create_viewport() {
    Viewport* vp = malloc(sizeof(Viewport));
    get_terminal_size(&vp->width, &vp->height);

    // Subtract 1 from height to prevent the last line from scrolling
    vp->height = vp->height > 1 ? vp->height - 1 : 24;
    if (vp->width < 20) vp->width = 80;

    // Allocate double buffers - each cell can hold multi-byte UTF-8
    vp->buffer = malloc(vp->height * sizeof(char*));
    vp->prev_buffer = malloc(vp->height * sizeof(char*));
    for (int i = 0; i < vp->height; i++) {
        vp->buffer[i] = malloc(vp->width * 4); // 4 bytes per cell for UTF-8
        vp->prev_buffer[i] = malloc(vp->width * 4);
        // Initialize both buffers to spaces
        for (int j = 0; j < vp->width; j++) {
            vp->buffer[i][j * 4] = ' ';
            vp->buffer[i][j * 4 + 1] = '\0';
            vp->prev_buffer[i][j * 4] = '\0'; // Mark as "never rendered"
            vp->prev_buffer[i][j * 4 + 1] = '\0';
        }
    }

    return vp;
}

void free_viewport(Viewport* vp) {
    for (int i = 0; i < vp->height; i++) {
        free(vp->buffer[i]);
        free(vp->prev_buffer[i]);
    }
    free(vp->buffer);
    free(vp->prev_buffer);
    free(vp);
}

void clear_buffer(Viewport* vp) {
    for (int j = 0; j < vp->height; j++) {
        for (int i = 0; i < vp->width; i++) {
            vp->buffer[j][i * 4] = ' ';
            vp->buffer[j][i * 4 + 1] = '\0';
        }
    }
}

void draw_pixel(Viewport* vp, int x, int y, const char* utf8) {
    if (x >= 0 && x < vp->width && y >= 0 && y < vp->height) {
        int idx = x * 4;
        int len = strlen(utf8);
        if (len > 3) len = 3;
        for (int i = 0; i < len; i++) {
            vp->buffer[y][idx + i] = utf8[i];
        }
        vp->buffer[y][idx + len] = '\0';
    }
}

void draw_pixel_char(Viewport* vp, int x, int y, char c) {
    if (x >= 0 && x < vp->width && y >= 0 && y < vp->height) {
        vp->buffer[y][x * 4] = c;
        vp->buffer[y][x * 4 + 1] = '\0';
    }
}

void draw_line(Viewport* vp, int x0, int y0, int x1, int y1, const char* utf8) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (1) {
        draw_pixel(vp, x0, y0, utf8);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void draw_circle(Viewport* vp, int cx, int cy, int r, const char* utf8) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_pixel(vp, cx + x, cy + y, utf8);
        draw_pixel(vp, cx - x, cy + y, utf8);
        draw_pixel(vp, cx + x, cy - y, utf8);
        draw_pixel(vp, cx - x, cy - y, utf8);
        draw_pixel(vp, cx + y, cy + x, utf8);
        draw_pixel(vp, cx - y, cy + x, utf8);
        draw_pixel(vp, cx + y, cy - x, utf8);
        draw_pixel(vp, cx - y, cy - x, utf8);

        x++;
        if (d > 0) {
            y--;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }
    }
}

void draw_filled_circle(Viewport* vp, int cx, int cy, int r, const char* utf8) {
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            if (x * x + y * y <= r * r) {
                draw_pixel(vp, cx + x, cy + y, utf8);
            }
        }
    }
}

void draw_string(Viewport* vp, int x, int y, const char* str) {
    int len = strlen(str);
    for (int i = 0; i < len && x + i < vp->width; i++) {
        draw_pixel_char(vp, x + i, y, str[i]);
    }
}

void draw_box(Viewport* vp, int x1, int y1, int x2, int y2, const char* style) {
    // Unicode box drawing characters
    const char* tl = (strcmp(style, "double") == 0) ? "╔" : "┌";
    const char* tr = (strcmp(style, "double") == 0) ? "╗" : "┐";
    const char* bl = (strcmp(style, "double") == 0) ? "╚" : "└";
    const char* br = (strcmp(style, "double") == 0) ? "╝" : "┘";
    const char* h = (strcmp(style, "double") == 0) ? "═" : "─";
    const char* v = (strcmp(style, "double") == 0) ? "║" : "│";

    // Corners
    draw_pixel(vp, x1, y1, tl);
    draw_pixel(vp, x2, y1, tr);
    draw_pixel(vp, x1, y2, bl);
    draw_pixel(vp, x2, y2, br);

    // Horizontal lines
    for (int x = x1 + 1; x < x2; x++) {
        draw_pixel(vp, x, y1, h);
        draw_pixel(vp, x, y2, h);
    }

    // Vertical lines
    for (int y = y1 + 1; y < y2; y++) {
        draw_pixel(vp, x1, y, v);
        draw_pixel(vp, x2, y, v);
    }
}

void render_buffer(Viewport* vp) {
    // Hide cursor during rendering
    printf("\033[?25l");

    for (int j = 0; j < vp->height; j++) {
        for (int i = 0; i < vp->width; i++) {
            int idx = i * 4;

            // Check if this cell changed
            if (strcmp(&vp->buffer[j][idx], &vp->prev_buffer[j][idx]) != 0) {
                // Move cursor to position (1-based ANSI coordinates)
                printf("\033[%d;%dH", j + 1, i + 1);

                // Print the character(s)
                if (vp->buffer[j][idx] != '\0') {
                    printf("%s", &vp->buffer[j][idx]);
                } else {
                    printf(" ");
                }

                // Copy to prev buffer
                strcpy(&vp->prev_buffer[j][idx], &vp->buffer[j][idx]);
            }
        }
    }

    // Show cursor again, move to bottom
    printf("\033[?25h");
    fflush(stdout);
}

void sleep_ms(int ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
}
