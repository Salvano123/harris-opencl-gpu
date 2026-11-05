#include "HarrisDetectorCL.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    std::string imgPath = (argc > 1) ? argv[1] : "lisc_high_res.jpg";
    cv::Mat fullImg = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (fullImg.empty()) {
        std::cerr << "Blad wczytywania obrazu: " << imgPath << "\n";
        return -1;
    }

    const int TILE_SIZE = 2048; // Rozmiar logiczny kafelka
    const int MARGIN = 7;       // Zakładka: 1 (Sobel) + 3 (Gauss) + 3 (NMS)
    const int MAX_TILE_DIM = TILE_SIZE + 2 * MARGIN;

    HarrisDetectorCL detector("kernels/harris_gpu.cl", MAX_TILE_DIM, MAX_TILE_DIM, 3);
    ProfilingData prof;
    std::vector<Vertex> allVertices;

    int fullW = fullImg.cols;
    int fullH = fullImg.rows;

    // Przetwarzanie kafelkowe (dla obrazów mniejszych niż 2048x2048 wykona się 1 iteracja)
    for (int tileY = 0; tileY < fullH; tileY += TILE_SIZE) {
        for (int tileX = 0; tileX < fullW; tileX += TILE_SIZE) {
            // Wyznaczenie współrzędnych kafelka wraz z zakładką (margin)
            int x0 = std::max(0, tileX - MARGIN);
            int y0 = std::max(0, tileY - MARGIN);
            int x1 = std::min(fullW, tileX + TILE_SIZE + MARGIN);
            int y1 = std::min(fullH, tileY + TILE_SIZE + MARGIN);

            // Wycięcie podobszaru i zapewnienie ciągłości pamięci (clone)
            cv::Mat tile = fullImg(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();

            auto tileVertices = detector.detect(tile.data, tile.cols, tile.rows, 0.05f, 1e-4f, prof);

            // Przeliczenie współrzędnych lokalnych kafelka na globalne obrazu
            for (const auto& v : tileVertices) {
                int globalX = x0 + v.x;
                int globalY = y0 + v.y;

                // Odrzucenie punktów z obszaru zakładki (aby nie dublować punktów na styku kafelków)
                if (globalX >= tileX && globalX < std::min(fullW, tileX + TILE_SIZE) &&
                    globalY >= tileY && globalY < std::min(fullH, tileY + TILE_SIZE)) {
                    allVertices.push_back({globalX, globalY});
                }
            }
        }
    }

    // Wyświetlenie raportu profilowania
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== RAPORT WYDAJNOSCI OPENCL (" << fullW << "x" << fullH << ") ===\n";
    std::cout << "Transfer Host -> Device : " << prof.hostToDeviceMs << " ms\n";
    std::cout << "Kernel 1 (Sobel 3x3)    : " << prof.sobelKernelMs << " ms\n";
    std::cout << "Kernel 2A (Gauss 1D X)  : " << prof.gaussXKernelMs << " ms\n";
    std::cout << "Kernel 2B (Gauss Y + H) : " << prof.gaussYHarrisKernelMs << " ms\n";
    std::cout << "Kernel 3 (NMS + Atomic) : " << prof.nmsKernelMs << " ms\n";
    std::cout << "Transfer Device -> Host : " << prof.deviceToHostMs << " ms\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "Laczny czas obliczen GPU: " << prof.totalGpuMs() << " ms\n";
    std::cout << "Wykryto wierzcholkow    : " << allVertices.size() << "\n";

    // Zapis wyniku do pliku
    cv::Mat outColor;
    cv::cvtColor(fullImg, outColor, cv::COLOR_GRAY2BGR);
    for (const auto& v : allVertices) {
        cv::circle(outColor, cv::Point(v.x, v.y), 3, cv::Scalar(0, 0, 255), -1);
    }
    cv::imwrite("wynik_harris.jpg", outColor);

    return 0;
}