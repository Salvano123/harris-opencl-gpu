#include "HarrisDetectorCL.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>

HarrisDetectorCL::HarrisDetectorCL(const std::string& kernelPath, int maxTileWidth, int maxTileHeight, int gaussRadius)
    : maxW(maxTileWidth), maxH(maxTileHeight), radius(gaussRadius) {
    initOpenCL(kernelPath);
    allocateBuffers(1.5f);
}

HarrisDetectorCL::~HarrisDetectorCL() {
    if (d_in) clReleaseMemObject(d_in);
    if (d_grad) clReleaseMemObject(d_grad);
    if (d_temp) clReleaseMemObject(d_temp);
    if (d_H) clReleaseMemObject(d_H);
    if (d_gauss) clReleaseMemObject(d_gauss);
    if (d_corners) clReleaseMemObject(d_corners);
    if (d_count) clReleaseMemObject(d_count);

    if (kSobel) clReleaseKernel(kSobel);
    if (kGaussX) clReleaseKernel(kGaussX);
    if (kGaussY) clReleaseKernel(kGaussY);
    if (kNms) clReleaseKernel(kNms);

    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

double HarrisDetectorCL::getEventMs(cl_event evt) {
    clWaitForEvents(1, &evt);
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    clReleaseEvent(evt);
    return (double)(end - start) * 1e-6; // Konwersja nanosekund na milisekundy
}

void HarrisDetectorCL::initOpenCL(const std::string& kernelPath) {
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    char devName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devName), devName, nullptr);
    std::cout << "[OpenCL] Urzadzenie GPU: " << devName << "\n";

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    std::ifstream file(kernelPath);
    if (!file.is_open()) throw std::runtime_error("Nie mozna otworzyc pliku kernela: " + kernelPath);
    std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* srcPtr = src.c_str();

    program = clCreateProgramWithSource(context, 1, &srcPtr, nullptr, &err);
    if (clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr) != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        throw std::runtime_error("Blad kompilacji OpenCL:\n" + std::string(log.data()));
    }

    kSobel  = clCreateKernel(program, "sobel_derivatives", &err);
    kGaussX = clCreateKernel(program, "gaussian_pass_x", &err);
    kGaussY = clCreateKernel(program, "gaussian_pass_y_and_harris", &err);
    kNms    = clCreateKernel(program, "nms_and_compact", &err);
}

void HarrisDetectorCL::allocateBuffers(float sigma) {
    cl_int err;
    size_t maxPixels = (size_t)maxW * maxH;

    // Wygenerowanie maski Gaussa 1D
    int size = 2 * radius + 1;
    std::vector<float> gauss(size);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        gauss[i + radius] = std::exp(-(i * i) / (2.0f * sigma * sigma));
        sum += gauss[i + radius];
    }
    for (float& v : gauss) v /= sum;

    d_in      = clCreateBuffer(context, CL_MEM_READ_ONLY, maxPixels * sizeof(cl_uchar), nullptr, &err);
    d_grad    = clCreateBuffer(context, CL_MEM_READ_WRITE, maxPixels * sizeof(cl_float4), nullptr, &err);
    d_temp    = clCreateBuffer(context, CL_MEM_READ_WRITE, maxPixels * sizeof(cl_float4), nullptr, &err);
    d_H       = clCreateBuffer(context, CL_MEM_READ_WRITE, maxPixels * sizeof(float), nullptr, &err);
    d_gauss   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), gauss.data(), &err);
    d_corners = clCreateBuffer(context, CL_MEM_WRITE_ONLY, maxCorners * sizeof(cl_int2), nullptr, &err);
    d_count   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
}

std::vector<Vertex> HarrisDetectorCL::detect(const unsigned char* grayData, int width, int height, 
                                             float k, float threshold, ProfilingData& prof) {
    size_t numPixels = (size_t)width * height;
    int zero = 0;

    cl_event evWrite, evSobel, evGx, evGy, evNms, evRead;

    // 1. Przesłanie obrazu i wyzerowanie licznika wierzchołków
    clEnqueueWriteBuffer(queue, d_count, CL_FALSE, 0, sizeof(int), &zero, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, d_in, CL_FALSE, 0, numPixels * sizeof(cl_uchar), grayData, 0, nullptr, &evWrite);

    size_t local[2] = {16, 16};
    size_t global[2] = {
        ((size_t)(width + 15) / 16) * 16,
        ((size_t)(height + 15) / 16) * 16
    };

    // 2. Kernel 1: Sobel
    clSetKernelArg(kSobel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kSobel, 1, sizeof(cl_mem), &d_grad);
    clSetKernelArg(kSobel, 2, sizeof(int), &width);
    clSetKernelArg(kSobel, 3, sizeof(int), &height);
    clEnqueueNDRangeKernel(queue, kSobel, 2, nullptr, global, local, 0, nullptr, &evSobel);

    // 3. Kernel 2A: Gauss X
    clSetKernelArg(kGaussX, 0, sizeof(cl_mem), &d_grad);
    clSetKernelArg(kGaussX, 1, sizeof(cl_mem), &d_temp);
    clSetKernelArg(kGaussX, 2, sizeof(cl_mem), &d_gauss);
    clSetKernelArg(kGaussX, 3, sizeof(int), &width);
    clSetKernelArg(kGaussX, 4, sizeof(int), &height);
    clSetKernelArg(kGaussX, 5, sizeof(int), &radius);
    clEnqueueNDRangeKernel(queue, kGaussX, 2, nullptr, global, local, 0, nullptr, &evGx);

    // 4. Kernel 2B: Gauss Y + odpowiedź Harrisa
    clSetKernelArg(kGaussY, 0, sizeof(cl_mem), &d_temp);
    clSetKernelArg(kGaussY, 1, sizeof(cl_mem), &d_H);
    clSetKernelArg(kGaussY, 2, sizeof(cl_mem), &d_gauss);
    clSetKernelArg(kGaussY, 3, sizeof(int), &width);
    clSetKernelArg(kGaussY, 4, sizeof(int), &height);
    clSetKernelArg(kGaussY, 5, sizeof(int), &radius);
    clSetKernelArg(kGaussY, 6, sizeof(float), &k);
    clEnqueueNDRangeKernel(queue, kGaussY, 2, nullptr, global, local, 0, nullptr, &evGy);

    // 5. Kernel 3: NMS + kompaktowanie
    clSetKernelArg(kNms, 0, sizeof(cl_mem), &d_H);
    clSetKernelArg(kNms, 1, sizeof(cl_mem), &d_corners);
    clSetKernelArg(kNms, 2, sizeof(cl_mem), &d_count);
    clSetKernelArg(kNms, 3, sizeof(int), &width);
    clSetKernelArg(kNms, 4, sizeof(int), &height);
    clSetKernelArg(kNms, 5, sizeof(float), &threshold);
    clSetKernelArg(kNms, 6, sizeof(int), &maxCorners);
    clEnqueueNDRangeKernel(queue, kNms, 2, nullptr, global, local, 0, nullptr, &evNms);

    // 6. Pobranie liczby znalezionych punktów
    int count = 0;
    clEnqueueReadBuffer(queue, d_count, CL_TRUE, 0, sizeof(int), &count, 0, nullptr, &evRead);
    count = std::min(count, maxCorners);

    std::vector<Vertex> results(count);
    if (count > 0) {
        cl_event evReadPts;
        clEnqueueReadBuffer(queue, d_corners, CL_TRUE, 0, count * sizeof(Vertex), results.data(), 0, nullptr, &evReadPts);
        prof.deviceToHostMs += getEventMs(evReadPts);
    }

    prof.hostToDeviceMs       += getEventMs(evWrite);
    prof.sobelKernelMs        += getEventMs(evSobel);
    prof.gaussXKernelMs       += getEventMs(evGx);
    prof.gaussYHarrisKernelMs += getEventMs(evGy);
    prof.nmsKernelMs          += getEventMs(evNms);
    prof.deviceToHostMs       += getEventMs(evRead);

    return results;
}