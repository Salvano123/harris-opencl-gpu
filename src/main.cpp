#include "HarrisDetectorCL.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
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

cv::Mat prepareScan(const cv::Mat& gray) {
    cv::Mat closed;
    const cv::Mat closeKernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(gray, closed, cv::MORPH_CLOSE, closeKernel);

    cv::Mat dark;
    cv::threshold(closed, dark, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    cv::Mat labels, stats, centroids;
    const int componentCount =
        cv::connectedComponentsWithStats(dark, labels, stats, centroids, 8, CV_32S);
    cv::Mat leafMask = cv::Mat::zeros(dark.size(), CV_8U);
    if (componentCount > 1) {
        int largest = 1;
        int largestArea = stats.at<int>(1, cv::CC_STAT_AREA);
        for (int i = 2; i < componentCount; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > largestArea) {
                largestArea = area;
                largest = i;
            }
        }
        const int minPartArea = std::max(1, static_cast<int>(largestArea * 0.001));
        for (int i = 1; i < componentCount; ++i) {
            if (stats.at<int>(i, cv::CC_STAT_AREA) >= minPartArea) {
                leafMask.setTo(255, labels == i);
            }
        }
    }

    const cv::Mat growKernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(leafMask, leafMask, growKernel);

    closed.setTo(255, leafMask == 0);
    return closed;
}

cv::Mat makePreview(const cv::Mat& fullImg, const std::vector<Vertex>& vertices) {
    cv::Mat outColor;
    cv::cvtColor(fullImg, outColor, cv::COLOR_GRAY2BGR);
    for (const auto& v : vertices) {
        cv::circle(outColor, cv::Point(v.x, v.y), 30, cv::Scalar(0, 0, 255), 5);
    }
    return outColor;
}

}

const char* kImagePath = "../leaf_02/l2nr002.tif";
// const char* kImagePath = "../BG_0005_2.tif";
constexpr float kThreshold = 0.03f;

int main() {
    const std::string imgPath = kImagePath;
    const float threshold = kThreshold;

    cv::Mat fullImg = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (fullImg.empty()) {
        std::cerr << "Blad wczytywania obrazu: " << imgPath << "\n";
        return -1;
    }

    // Podglad kreslimy na surowym skanie. Detekcja dostaje obraz po filtrze.
    const cv::Mat sourceForPreview = fullImg.clone();
    fullImg = prepareScan(fullImg);

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

    const cv::Mat preview = makePreview(sourceForPreview, allVertices);
    const auto slash = imgPath.find_last_of("/\\");
    const std::string fileName = imgPath.substr(slash == std::string::npos ? 0 : slash + 1);
    const auto dot = fileName.find_last_of('.');
    const std::string stem = fileName.substr(0, dot == std::string::npos ? fileName.size() : dot);
    if (!cv::imwrite("wyniki/" + stem + "_harris.jpg", preview)) {
        std::cerr << "Nie zapisano podgladu: wyniki/" << stem << "_harris.jpg\n";
    }
    cv::namedWindow("Harris Detection", cv::WINDOW_NORMAL);
    cv::imshow("Harris Detection", preview);
    // cv::imwrite("harris_result.jpg", preview);

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
    std::cout << "Nacisnij SPACJE w oknie obrazu, aby zakonczyc.\n";

    while (cv::waitKey(0) != ' ') {}

    return 0;
}
