#pragma once
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <vector>
#include <string>

struct Vertex {
    int x, y;
};

struct ProfilingData {
    double hostToDeviceMs = 0.0;
    double sobelKernelMs = 0.0;
    double gaussXKernelMs = 0.0;
    double gaussYHarrisKernelMs = 0.0;
    double nmsKernelMs = 0.0;
    double deviceToHostMs = 0.0;
    
    double totalGpuMs() const {
        return sobelKernelMs + gaussXKernelMs + gaussYHarrisKernelMs + nmsKernelMs;
    }
};

class HarrisDetectorCL {
public:
    HarrisDetectorCL(const std::string& kernelPath, int maxTileWidth, int maxTileHeight, int gaussRadius = 3);
    ~HarrisDetectorCL();

    // Przetwarza pojedynczy obraz lub kafelek (tile)
    std::vector<Vertex> detect(const unsigned char* grayData, int width, int height, 
                               float k, float threshold, ProfilingData& prof);

private:
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    cl_kernel kSobel = nullptr;
    cl_kernel kGaussX = nullptr;
    cl_kernel kGaussY = nullptr;
    cl_kernel kNms = nullptr;

    // Prealokowane bufory (unikanie alokacji w pętli!)
    cl_mem d_in = nullptr;
    cl_mem d_grad = nullptr;
    cl_mem d_temp = nullptr;
    cl_mem d_H = nullptr;
    cl_mem d_gauss = nullptr;
    cl_mem d_corners = nullptr;
    cl_mem d_count = nullptr;

    int maxW, maxH, radius;
    int maxCorners = 200000;

    void initOpenCL(const std::string& kernelPath);
    void allocateBuffers(float sigma);
    static double getEventMs(cl_event evt);
};