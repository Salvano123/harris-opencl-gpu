#include "HarrisDetectorCL.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace {

struct TileGeom {
    int tileX;
    int tileY;
    int x0;
    int y0;
    int width;
    int height;
};

std::vector<TileGeom> makeTiles(int fullW, int fullH, int tileSize, int margin) {
    std::vector<TileGeom> tiles;
    for (int tileY = 0; tileY < fullH; tileY += tileSize) {
        for (int tileX = 0; tileX < fullW; tileX += tileSize) {
            const int x0 = std::max(0, tileX - margin);
            const int y0 = std::max(0, tileY - margin);
            const int x1 = std::min(fullW, tileX + tileSize + margin);
            const int y1 = std::min(fullH, tileY + tileSize + margin);
            tiles.push_back({tileX, tileY, x0, y0, x1 - x0, y1 - y0});
        }
    }
    return tiles;
}

std::vector<Vertex> collectVertices(HarrisDetectorCL& detector, const cv::Mat& fullImg,
                                    const std::vector<TileGeom>& tiles, int tileSize,
                                    float k, float threshold, ProfilingData& prof) {
    std::vector<Vertex> allVertices;
    const int fullW = fullImg.cols;
    const int fullH = fullImg.rows;

    for (const TileGeom& tileGeom : tiles) {
        cv::Mat tile = fullImg(cv::Rect(tileGeom.x0, tileGeom.y0, tileGeom.width, tileGeom.height)).clone();
        const auto tileVertices = detector.detect(tile.data, tile.cols, tile.rows, k, threshold, prof);

        for (const auto& v : tileVertices) {
            const int globalX = tileGeom.x0 + v.x;
            const int globalY = tileGeom.y0 + v.y;
            if (globalX >= tileGeom.tileX && globalX < std::min(fullW, tileGeom.tileX + tileSize) &&
                globalY >= tileGeom.tileY && globalY < std::min(fullH, tileGeom.tileY + tileSize)) {
                allVertices.push_back({globalX, globalY});
            }
        }
    }
    return allVertices;
}

bool writePreview(const cv::Mat& fullImg, const std::vector<Vertex>& vertices, const std::string& path) {
    cv::Mat outColor;
    cv::cvtColor(fullImg, outColor, cv::COLOR_GRAY2BGR);
    for (const auto& v : vertices) {
        cv::circle(outColor, cv::Point(v.x, v.y), 3, cv::Scalar(0, 0, 255), -1);
    }
    return cv::imwrite(path, outColor);
}

bool parseThreshold(const char* text, float& out) {
    char* end = nullptr;
    out = std::strtof(text, &end);
    if (end == text || (end != nullptr && *end != '\0')) return false;
    return std::isfinite(out) && out >= 0.f;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Uzycie: harris_app.exe <obraz> [prog]\n";
        return 1;
    }

    const std::string imgPath = argv[1];
    // Staly prog dobrany recznie (podglad l2nr001). Nie jest liczony z obrazu.
    constexpr float kDefaultThreshold = 0.02f;
    float threshold = kDefaultThreshold;
    if (argc == 3 && !parseThreshold(argv[2], threshold)) {
        std::cerr << "Niepoprawny prog: " << argv[2] << "\n";
        return 1;
    }

    cv::Mat fullImg = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (fullImg.empty()) {
        std::cerr << "Blad wczytywania obrazu: " << imgPath << "\n";
        return -1;
    }

    const int TILE_SIZE = 2048; // Rozmiar logiczny kafelka
    const int MARGIN = 7;       // Zakładka: 1 (Sobel) + 3 (Gauss) + 3 (NMS)
    const int MAX_TILE_DIM = TILE_SIZE + 2 * MARGIN;
    const float kHarris = 0.05f;

    HarrisDetectorCL detector("kernels/harris_gpu.cl", MAX_TILE_DIM, MAX_TILE_DIM, 3);
    const int fullW = fullImg.cols;
    const int fullH = fullImg.rows;
    const std::vector<TileGeom> tiles = makeTiles(fullW, fullH, TILE_SIZE, MARGIN);

    ProfilingData prof;
    const std::vector<Vertex> allVertices =
        collectVertices(detector, fullImg, tiles, TILE_SIZE, kHarris, threshold, prof);

    const std::filesystem::path outDir = "wyniki";
    std::filesystem::create_directories(outDir);
    const std::filesystem::path outPath =
        outDir / (std::filesystem::path(imgPath).stem().string() + "_harris.jpg");
    if (!writePreview(fullImg, allVertices, outPath.string())) {
        std::cerr << "Blad zapisu obrazu: " << outPath.string() << "\n";
        return -1;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== RAPORT WYDAJNOSCI OPENCL (" << fullW << "x" << fullH << ") ===\n";
    std::cout << "Prog                    : " << std::setprecision(6) << threshold << std::setprecision(3) << "\n";
    std::cout << "k                       : " << kHarris << "\n";
    std::cout << "Transfer Host -> Device : " << prof.hostToDeviceMs << " ms\n";
    std::cout << "Kernel 1 (Sobel 3x3)    : " << prof.sobelKernelMs << " ms\n";
    std::cout << "Kernel 2A (Gauss 1D X)  : " << prof.gaussXKernelMs << " ms\n";
    std::cout << "Kernel 2B (Gauss Y + H) : " << prof.gaussYHarrisKernelMs << " ms\n";
    std::cout << "Kernel 3 (NMS + Atomic) : " << prof.nmsKernelMs << " ms\n";
    std::cout << "Transfer Device -> Host : " << prof.deviceToHostMs << " ms\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "Laczny czas obliczen GPU: " << prof.totalGpuMs() << " ms\n";
    std::cout << "Wykryto wierzcholkow    : " << allVertices.size() << "\n";
    std::cout << "Zapisano                : " << outPath.string() << "\n";
    return 0;
}
