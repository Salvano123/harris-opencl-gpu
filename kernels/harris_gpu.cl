// Kernel 1: Konwersja uchar -> float + Sobel 3x3 + pakowanie do float4 (Ix^2, Iy^2, Ixy, 0)
__kernel void sobel_derivatives(
    __global const uchar* input_gray,
    __global float4* grad_out,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        grad_out[idx] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Odczyt i normalizacja do [0.0, 1.0] w locie
    float p00 = input_gray[(y - 1) * width + (x - 1)] * (1.0f / 255.0f);
    float p01 = input_gray[(y - 1) * width + x]       * (1.0f / 255.0f);
    float p02 = input_gray[(y - 1) * width + (x + 1)] * (1.0f / 255.0f);

    float p10 = input_gray[y * width + (x - 1)]       * (1.0f / 255.0f);
    float p12 = input_gray[y * width + (x + 1)]       * (1.0f / 255.0f);

    float p20 = input_gray[(y + 1) * width + (x - 1)] * (1.0f / 255.0f);
    float p21 = input_gray[(y + 1) * width + x]       * (1.0f / 255.0f);
    float p22 = input_gray[(y + 1) * width + (x + 1)] * (1.0f / 255.0f);

    float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
    float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

    grad_out[idx] = (float4)(gx * gx, gy * gy, gx * gy, 0.0f);
}

// Kernel 2A: Separowalny filtr Gaussa - przebieg poziomy (1D)
__kernel void gaussian_pass_x(
    __global const float4* grad_in,
    __global float4* temp_out,
    __constant float* gauss_1d,
    const int width,
    const int height,
    const int radius)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    if (x < radius || x >= width - radius) {
        temp_out[y * width + x] = (float4)(0.0f);
        return;
    }

    float4 sum = (float4)(0.0f);
    int row_offset = y * width;

    for (int dx = -radius; dx <= radius; dx++) {
        sum += grad_in[row_offset + x + dx] * gauss_1d[dx + radius];
    }

    temp_out[row_offset + x] = sum;
}

// Kernel 2B: Separowalny filtr Gaussa - przebieg pionowy (1D) + odpowiedź Harrisa H(x,y)
__kernel void gaussian_pass_y_and_harris(
    __global const float4* temp_in,
    __global float* H_out,
    __constant float* gauss_1d,
    const int width,
    const int height,
    const int radius,
    const float k)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    if (x < radius || x >= width - radius || y < radius || y >= height - radius) {
        H_out[y * width + x] = 0.0f;
        return;
    }

    float4 sum = (float4)(0.0f);
    for (int dy = -radius; dy <= radius; dy++) {
        sum += temp_in[(y + dy) * width + x] * gauss_1d[dy + radius];
    }

    // sum.x = <Ix^2>, sum.y = <Iy^2>, sum.z = <Ixy>
    float detM = (sum.x * sum.y) - (sum.z * sum.z);
    float traceM = sum.x + sum.y;

    H_out[y * width + x] = detM - k * (traceM * traceM);
}

// Kernel 3: Non-Maximum Suppression z pamięcią __local i atomowym zapisem listy wierzchołków
#define WG_SIZE 16
#define NMS_RADIUS 3
#define TILE_DIM (WG_SIZE + 2 * NMS_RADIUS)

__kernel void nms_and_compact(
    __global const float* H_in,
    __global int2* corners_list,      // Wyjściowa tablica współrzędnych (x, y)
    volatile __global int* corner_count, // Licznik znalezionych punktów
    const int width,
    const int height,
    const float threshold,
    const int max_corners)
{
    __local float tile[TILE_DIM][TILE_DIM];

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_group_id(0) * WG_SIZE + lx;
    int gy = get_group_id(1) * WG_SIZE + ly;

    // 1. Współbieżne ładowanie kafelka wraz z ramką (halo/apron) do pamięci __local
    for (int j = ly; j < TILE_DIM; j += WG_SIZE) {
        int img_y = clamp((int)(get_group_id(1) * WG_SIZE + j - NMS_RADIUS), 0, height - 1);
        for (int i = lx; i < TILE_DIM; i += WG_SIZE) {
            int img_x = clamp((int)(get_group_id(0) * WG_SIZE + i - NMS_RADIUS), 0, width - 1);
            tile[j][i] = H_in[img_y * width + img_x];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx < NMS_RADIUS || gx >= width - NMS_RADIUS || 
        gy < NMS_RADIUS || gy >= height - NMS_RADIUS) {
        return;
    }

    int tx = lx + NMS_RADIUS;
    int ty = ly + NMS_RADIUS;
    float current_H = tile[ty][tx];

    // 2. Sprawdzenie progu i lokalnego maksimum w pamięci lokalnej
    if (current_H > threshold) {
        bool is_max = true;
        for (int dy = -NMS_RADIUS; dy <= NMS_RADIUS; dy++) {
            for (int dx = -NMS_RADIUS; dx <= NMS_RADIUS; dx++) {
                if (dx == 0 && dy == 0) continue;
                if (tile[ty + dy][tx + dx] >= current_H) {
                    is_max = false;
                    break;
                }
            }
            if (!is_max) break;
        }

        // 3. Atomowe dopisanie wierzchołka do tablicy wynikowej
        if (is_max) {
            int out_idx = atomic_inc(corner_count);
            if (out_idx < max_corners) {
                corners_list[out_idx] = (int2)(gx, gy);
            }
        }
    }
}