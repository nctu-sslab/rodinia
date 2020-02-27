#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "timer.h"

#ifdef OMP_AT
#include "fake_at_runtime.h"
#endif

void run(int argc, char **argv);

/* define timer macros */
#define pin_stats_reset() startCycle()
#define pin_stats_pause(cycles) stopCycle(cycles)
#define pin_stats_dump(cycles) printf("timer: %Lu\n", cycles)

int rows, cols;
int *data;
int **wall;
int *result;

void init(int argc, char **argv) {
    if (argc == 3) {
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
    } else {
        printf("Usage: pathfiner width num_of_steps\n");
        exit(0);
    }
    data = new int[rows * cols];
    wall = new int *[rows];
    for (int n = 0; n < rows; n++)
        wall[n] = data + cols * n;
    result = new int[cols];

    srand(7);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            wall[i][j] = rand() % 10;
        }
    }
    for (int j = 0; j < cols; j++)
        result[j] = wall[0][j];

    if (getenv("OUTPUT")) {
        FILE *file = fopen("output.txt", "w");

        fprintf(file, "wall:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(file, "%d ", wall[i][j]);
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

int main(int argc, char **argv) {
    run(argc, argv);

    return EXIT_SUCCESS;
}

void run(int argc, char **argv) {
    init(argc, argv);

    unsigned long long cycles;

    int *src, *dst, *temp;
    int min;

    dst = result;
    src = new int[cols];

    pin_stats_reset();
#ifdef OMP_OFFLOAD
#pragma omp target enter data map (to: src[:cols], wall[:rows], dst[:cols])
    for (int i = 0; i < rows; i++) {
      #pragma omp target enter data map(to: wall[i][:cols])
    }
#endif
    printf("Rows: %d\n", rows);
    for (int t = 0; t < rows - 1; t++) {
        temp = src;
        src = dst;
        dst = temp;
#ifdef OMP_OFFLOAD
#ifdef OMP_AT
        FAKE_AT_RUNTIME_PRE_KERNEL;
#endif
#pragma omp target teams distribute parallel for private(min)
#else
#pragma omp parallel for private(min)
#endif
        for (int n = 0; n < cols; n++) {
            min = src[n];
            if (n > 0)
                min = MIN(min, src[n - 1]);
            if (n < cols - 1)
                min = MIN(min, src[n + 1]);
#ifdef OMP_AT
            dst[n] = AT(wall[t + 1])[n] + min;
#else
            dst[n] = wall[t + 1][n] + min;
#endif
          //  tmp[n] = AT(wall[t+1]+n);
        }
    //printf("wall: %d + min(src %d, %d) = dst[0]: %d -> tmp: %p vs %p\n", wall[t+1][0], src[0], src[1], dst[0], tmp[0], wall[t+1]);
//    unsetenv("TABLE");
//#define NNN 3
    //printf("%p vs %p\n", tmp[NNN], wall[t+1]+NNN);
    }
#ifdef OMP_OFFLOAD
    // retrieve data
#pragma omp target exit data map (from: dst[:cols])
#endif

 //   printf("dst[0]: %d\n", dst[0]);

    pin_stats_pause(cycles);
    pin_stats_dump(cycles);

    if (getenv("OUTPUT")) {
        FILE *file = fopen("output.txt", "a");

        fprintf(file, "data:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", data[i]);
        fprintf(file, "\n");

        fprintf(file, "result:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", dst[i]);
        fprintf(file, "\n");

        fclose(file);
    }

    delete[] data;
    delete[] wall;
    delete[] dst;
    delete[] src;
}
