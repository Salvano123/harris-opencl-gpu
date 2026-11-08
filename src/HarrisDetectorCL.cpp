#include "HarrisDetectorCL.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>

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

namespace {

void checkCl(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed, cl_int=" + std::to_string(err));
    }
}

std::string clString(cl_platform_id platform, cl_platform_info info) {
    size_t size = 0;
    if (clGetPlatformInfo(platform, info, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::vector<char> buf(size);
    clGetPlatformInfo(platform, info, size, buf.data(), nullptr);
    return std::string(buf.data());
}

std::string clString(cl_device_id device, cl_device_info info) {
    size_t size = 0;
    if (clGetDeviceInfo(device, info, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::vector<char> buf(size);
    clGetDeviceInfo(device, info, size, buf.data(), nullptr);
    return std::string(buf.data());
}

std::string toLower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

int deviceRank(cl_device_type type, const std::string& platformName, const std::string& deviceName) {
    const std::string blob = toLower(platformName + " " + deviceName);
    const bool gpu = (type & CL_DEVICE_TYPE_GPU) != 0;
    if (gpu && blob.find("nvidia") != std::string::npos) return 0;
    if (gpu && blob.find("iris") != std::string::npos) return 1;
    if (gpu && blob.find("intel") != std::string::npos) return 2;
    if (gpu) return 3;
    return 4;
}

cl_device_id selectDevice() {
    cl_uint numPlatforms = 0;
    checkCl(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs");
    if (numPlatforms == 0) throw std::runtime_error("Nie znaleziono zadnej platformy OpenCL");

    std::vector<cl_platform_id> platforms(numPlatforms);
    checkCl(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr), "clGetPlatformIDs");

    struct Candidate {
        cl_device_id device;
        int rank;
        std::string label;
    };
    std::vector<Candidate> candidates;

    std::cout << "[OpenCL] Platformy (" << numPlatforms << "):\n";
    for (cl_platform_id platform : platforms) {
        const std::string pname = clString(platform, CL_PLATFORM_NAME);
        std::cout << "  Platforma: " << pname << "\n";

        cl_uint numDevices = 0;
        cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        if (err == CL_DEVICE_NOT_FOUND || numDevices == 0) {
            std::cout << "    (brak urzadzen)\n";
            continue;
        }
        checkCl(err, "clGetDeviceIDs");

        std::vector<cl_device_id> devices(numDevices);
        checkCl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr), "clGetDeviceIDs");

        for (cl_device_id device : devices) {
            cl_device_type type = 0;
            checkCl(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr), "clGetDeviceInfo");
            const std::string dname = clString(device, CL_DEVICE_NAME);
            const char* typeStr = (type & CL_DEVICE_TYPE_GPU) ? "GPU" :
                                  (type & CL_DEVICE_TYPE_CPU) ? "CPU" : "OTHER";
            std::cout << "    [" << typeStr << "] " << dname << "\n";
            candidates.push_back({device, deviceRank(type, pname, dname), pname + " / " + dname});
        }
    }

    if (candidates.empty()) throw std::runtime_error("Nie znaleziono zadnego urzadzenia OpenCL");

    const auto best = std::min_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.rank < b.rank; });
    std::cout << "[OpenCL] Wybrane urzadzenie: " << best->label << "\n";
    return best->device;
}

} // namespace

void HarrisDetectorCL::initOpenCL(const std::string& kernelPath) {
    cl_int err = CL_SUCCESS;
    cl_device_id device = selectDevice();

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkCl(err, "clCreateContext");
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkCl(err, "clCreateCommandQueue");

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
    checkCl(err, "clCreateKernel(sobel_derivatives)");
    kGaussX = clCreateKernel(program, "gaussian_pass_x", &err);
    checkCl(err, "clCreateKernel(gaussian_pass_x)");
    kGaussY = clCreateKernel(program, "gaussian_pass_y_and_harris", &err);
    checkCl(err, "clCreateKernel(gaussian_pass_y_and_harris)");
    kNms    = clCreateKernel(program, "nms_and_compact", &err);
    checkCl(err, "clCreateKernel(nms_and_compact)");
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
    checkCl(err, "clCreateBuffer(d_in)");
    d_grad    = clCreateBuffer(context, CL_MEM_READ_WRITE, maxPixels * sizeof(cl_float4), nullptr, &err);
    checkCl(err, "clCreateBuffer(d_grad)");
    d_temp    = clCreateBuffer(context, CL_MEM_READ_WRITE, maxPixels * sizeof(cl_float4), nullptr, &err);
    checkCl(err, "clCreateBuffer(d_temp)");
    d_H       = clCreateBuffer(context, CL_MEM_READ_WRITE, maxPixels * sizeof(float), nullptr, &err);
    checkCl(err, "clCreateBuffer(d_H)");
    d_gauss   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), gauss.data(), &err);
    checkCl(err, "clCreateBuffer(d_gauss)");
    d_corners = clCreateBuffer(context, CL_MEM_WRITE_ONLY, maxCorners * sizeof(cl_int2), nullptr, &err);
    checkCl(err, "clCreateBuffer(d_corners)");
    d_count   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
    checkCl(err, "clCreateBuffer(d_count)");
}

void HarrisDetectorCL::enqueueResponse(const unsigned char* grayData, int width, int height, float k,
                                       cl_event& evWrite, cl_event& evSobel, cl_event& evGx, cl_event& evGy) {
    if (width <= 0 || height <= 0 || width > maxW || height > maxH) {
        throw std::runtime_error("Rozmiar kafelka przekracza prealokowany bufor");
    }

    const size_t numPixels = (size_t)width * height;
    checkCl(clEnqueueWriteBuffer(queue, d_in, CL_FALSE, 0, numPixels * sizeof(cl_uchar), grayData, 0, nullptr, &evWrite),
            "clEnqueueWriteBuffer(d_in)");

    const size_t local[2] = {16, 16};
    const size_t global[2] = {
        ((size_t)(width + 15) / 16) * 16,
        ((size_t)(height + 15) / 16) * 16
    };

    checkCl(clSetKernelArg(kSobel, 0, sizeof(cl_mem), &d_in), "clSetKernelArg");
    checkCl(clSetKernelArg(kSobel, 1, sizeof(cl_mem), &d_grad), "clSetKernelArg");
    checkCl(clSetKernelArg(kSobel, 2, sizeof(int), &width), "clSetKernelArg");
    checkCl(clSetKernelArg(kSobel, 3, sizeof(int), &height), "clSetKernelArg");
    checkCl(clEnqueueNDRangeKernel(queue, kSobel, 2, nullptr, global, local, 0, nullptr, &evSobel), "sobel");

    checkCl(clSetKernelArg(kGaussX, 0, sizeof(cl_mem), &d_grad), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussX, 1, sizeof(cl_mem), &d_temp), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussX, 2, sizeof(cl_mem), &d_gauss), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussX, 3, sizeof(int), &width), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussX, 4, sizeof(int), &height), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussX, 5, sizeof(int), &radius), "clSetKernelArg");
    checkCl(clEnqueueNDRangeKernel(queue, kGaussX, 2, nullptr, global, local, 0, nullptr, &evGx), "gaussian_pass_x");

    checkCl(clSetKernelArg(kGaussY, 0, sizeof(cl_mem), &d_temp), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussY, 1, sizeof(cl_mem), &d_H), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussY, 2, sizeof(cl_mem), &d_gauss), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussY, 3, sizeof(int), &width), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussY, 4, sizeof(int), &height), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussY, 5, sizeof(int), &radius), "clSetKernelArg");
    checkCl(clSetKernelArg(kGaussY, 6, sizeof(float), &k), "clSetKernelArg");
    checkCl(clEnqueueNDRangeKernel(queue, kGaussY, 2, nullptr, global, local, 0, nullptr, &evGy), "gaussian_pass_y_and_harris");
}

std::vector<Vertex> HarrisDetectorCL::detect(const unsigned char* grayData, int width, int height,
                                             float k, float threshold, ProfilingData& prof) {
    int zero = 0;
    checkCl(clEnqueueWriteBuffer(queue, d_count, CL_FALSE, 0, sizeof(int), &zero, 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(d_count)");

    cl_event evWrite = nullptr, evSobel = nullptr, evGx = nullptr, evGy = nullptr;
    enqueueResponse(grayData, width, height, k, evWrite, evSobel, evGx, evGy);

    const size_t local[2] = {16, 16};
    const size_t global[2] = {
        ((size_t)(width + 15) / 16) * 16,
        ((size_t)(height + 15) / 16) * 16
    };

    cl_event evNms = nullptr;
    checkCl(clSetKernelArg(kNms, 0, sizeof(cl_mem), &d_H), "clSetKernelArg");
    checkCl(clSetKernelArg(kNms, 1, sizeof(cl_mem), &d_corners), "clSetKernelArg");
    checkCl(clSetKernelArg(kNms, 2, sizeof(cl_mem), &d_count), "clSetKernelArg");
    checkCl(clSetKernelArg(kNms, 3, sizeof(int), &width), "clSetKernelArg");
    checkCl(clSetKernelArg(kNms, 4, sizeof(int), &height), "clSetKernelArg");
    checkCl(clSetKernelArg(kNms, 5, sizeof(float), &threshold), "clSetKernelArg");
    checkCl(clSetKernelArg(kNms, 6, sizeof(int), &maxCorners), "clSetKernelArg");
    checkCl(clEnqueueNDRangeKernel(queue, kNms, 2, nullptr, global, local, 0, nullptr, &evNms), "nms_and_compact");

    int count = 0;
    cl_event evRead = nullptr;
    checkCl(clEnqueueReadBuffer(queue, d_count, CL_TRUE, 0, sizeof(int), &count, 0, nullptr, &evRead),
            "clEnqueueReadBuffer(d_count)");
    count = std::min(count, maxCorners);

    std::vector<Vertex> results(count);
    if (count > 0) {
        cl_event evReadPts = nullptr;
        checkCl(clEnqueueReadBuffer(queue, d_corners, CL_TRUE, 0, count * sizeof(Vertex), results.data(), 0, nullptr, &evReadPts),
                "clEnqueueReadBuffer(d_corners)");
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