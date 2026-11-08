# Detekcja wierzchołków Harrisa w OpenCL

Program wykrywa wierzchołki metodą Harrisa na GPU. OpenCV służy do wczytania obrazu i zapisu podglądu.

## Wymagania

- CMake 3.15 lub nowszy
- kompilator C++17
- OpenCV
- OpenCL: nagłówki, biblioteka ICD oraz sterownik widoczny jako platforma OpenCL

Sprawdzona konfiguracja: Windows, generator `MinGW Makefiles`, g++ z MSYS2 (C++17), OpenCV 4.13 i loader OpenCL z tego samego prefiksu MinGW.

## Budowanie

W katalogu projektu:

```
cmake -G "MinGW Makefiles" -S . -B build -DCMAKE_PREFIX_PATH="<prefiks-mingw>" -DOpenCV_DIR="<katalog-z-OpenCVConfig.cmake>" -DOpenCL_INCLUDE_DIR="<katalog-include>" -DOpenCL_LIBRARY="<biblioteka-OpenCL>"
cmake --build build
```

Katalog `build/` to tylko lokalny wynik CMake i nie wchodzi do repozytorium. Po `cmake --build build` program wykonywalny leży w tym katalogu.

Znaczenie zmiennych:

- `CMAKE_PREFIX_PATH` — prefiks, w którym leżą OpenCV i OpenCL (w MSYS2 jest to katalog `mingw64`)
- `OpenCV_DIR` — katalog zawierający `OpenCVConfig.cmake`
- `OpenCL_INCLUDE_DIR` — katalog zawierający podkatalog `CL` (z plikiem `cl.h`)
- `OpenCL_LIBRARY` — biblioteka importu OpenCL używana przez ten kompilator (dla MinGW: `libOpenCL.dll.a`)

Przy uruchamianiu katalog `bin` tego prefiksu musi być w `PATH`, żeby znalazły się DLL-e OpenCV i OpenCL.

## Uruchomienie

CMake kopiuje `kernels/` do katalogu budowania. Program otwiera `kernels/harris_gpu.cl` względem bieżącego katalogu, więc odpalaj go z `build`:

```
cd build
harris_app.exe sciezka\do\obrazu.tif
harris_app.exe sciezka\do\obrazu.tif 0.02
```

Pierwszy argument to obraz. Drugi, opcjonalny, to próg odpowiedzi Harrisa (kropka jako separator dziesiętny). Bez niego program używa stałej `0.02`. Współczynnik k wynosi `0.05`. Próg nie jest wyliczany z obrazu; adaptacyjny dobór progu nie jest zaimplementowany.

Na starcie wypisywane są wszystkie platformy OpenCL i ich urządzenia. Jeśli jest karta NVIDIA, program wybiera ją. W przeciwnym razie bierze inną kartę GPU, a gdy GPU nie ma — dowolne urządzenie.

Podgląd zapisuje jako `wyniki/<nazwa>_harris.jpg` w bieżącym katalogu, osobno dla każdego pliku wejściowego. Na konsoli widać użyty próg, liczbę wierzchołków i czasy kerneli.
