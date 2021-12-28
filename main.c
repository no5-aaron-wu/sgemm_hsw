#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>

#ifdef __cplusplus
extern "C" {
#endif

void sgemm_kernel_x64_fma_m4n24(float *a,
    float *b,
    float *c,
    int m,
    int k);

void sgemm_kernel_x64_fma_m4n16(float *a,
    float *b,
    float *c,
    int m,
    int k);

#ifdef __cplusplus
}
#endif

static double get_time(struct timespec *start, struct timespec *end)
{
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

static void thread_bind(int cpu)
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu, &cpu_set);
    if (pthread_setaffinity_np(pthread_self(),
            sizeof(cpu_set_t), &cpu_set) != 0)
    {
        fprintf(stderr, "Error: cpu[%d] bind failed.\n", cpu);
        exit(0);
    }
}

static void *page_alloc(size_t size)
{
    void *data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    if (data == (void*)-1)
    {
        fprintf(stderr, "Error(MemData::Construction): mmap failed.\n");
        exit(0);
    }
    return data;
}

static void page_free(void *mem, size_t size)
{
    munmap(mem, size);
}

void save_bin(const char *file_name, float *rst, int num)
{
    FILE *fp = fopen(file_name, "wb");
    fwrite(rst, num, sizeof(float), fp);
    fclose(fp);
}

void sgemm_naive(float *a, float *b, float *c, int m, int n, int k)
{
    int i, j, kk;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (kk = 0; kk < k; kk++)
            {
                c[i * n + j] += a[i * k + kk] * b[kk * n + j];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int i;

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s m k 24/16\n", argv[0]);
        return -1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    if (n != 24 && n != 16){
        fprintf(stderr, "n must be 24 or 16 currently\n");
        return -1;
    }

    long comp = 2L * m * k * n;
    int loop_time = (int)(2e11 / comp);

    struct timespec start, end;
    double t, gflops;

    thread_bind(0);

    float *a = (float*)page_alloc(m * k * sizeof(float));
    float *b = (float*)page_alloc(k * n * sizeof(float));
    float *c1 = (float*)page_alloc(m * n * sizeof(float));
    float *c2 = (float*)page_alloc(m * n * sizeof(float));

    srand(time(NULL));

    for (i = 0; i < m * k; i++)
    {
        a[i] = (float)rand() / (float)RAND_MAX;
    }
    for (i = 0; i < k * n; i++)
    {
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    if (n == 24){
        // fma-tuned version
        // warm up
        for (i = 0; i < loop_time; i++)
        {
            sgemm_kernel_x64_fma_m4n24(a, b, c2, m, k);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        for (i = 0; i < loop_time; i++)
        {
            sgemm_kernel_x64_fma_m4n24(a, b, c2, m, k);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);    
    } else if (n == 16){
        // fma-tuned version
        // warm up
        for (i = 0; i < loop_time; i++)
        {
            sgemm_kernel_x64_fma_m4n16(a, b, c2, m, k);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        for (i = 0; i < loop_time; i++)
        {
            sgemm_kernel_x64_fma_m4n16(a, b, c2, m, k);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);    
    }
    

    t = get_time(&start, &end) / loop_time;
    gflops = (double)comp / t * 1e-9;

    printf("sgemm_kernel_x64_fma(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, 24, k, t * 1e6, gflops);

    memset(c1, 0, m * n * sizeof(float));
    memset(c2, 0, m * n * sizeof(float));
    sgemm_naive(a, b, c1, m, n, k);
    if (n == 24){
        sgemm_kernel_x64_fma_m4n24(a, b, c2, m, k);
    } else if (n == 16){
        sgemm_kernel_x64_fma_m4n16(a, b, c2, m, k);
    }
    
    save_bin("naive.bin", c1, m * n);
    save_bin("tuned.bin", c2, m * n);

    printf("sgemm_naive result: naive.bin\n");
    printf("sgemm_kernel_x64_fma_m4n24 result: tuned.bin\n");
    printf("Use fp_diff(https://github.com/pigirons/fp_diff) to compare the results.\n");

    page_free(a, m * k * sizeof(float));
    page_free(b, k * n * sizeof(float));
    page_free(c1, m * n * sizeof(float));
    page_free(c2, m * n * sizeof(float));
    
    return 0;
}

