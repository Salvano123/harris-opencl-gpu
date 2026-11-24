# Detekcja wierzchołków Harrisa w OpenCL

Program wykrywa wierzchołki metodą Harrisa na GPU. OpenCV służy do wczytania obrazu i zapisu podglądu. Przygotowanie tła obrazu wykonuje się na CPU, zanim ruszą kernele Harrisa na GPU.

## Wymagania

- CMake 3.15 lub nowszy
- kompilator C++17
- OpenCV
- OpenCL: nagłówki, biblioteka ICD oraz sterownik widoczny jako platforma OpenCL

Sprawdzona konfiguracja: Windows, generator `MinGW Makefiles`, g++ z MSYS2 (C++17), OpenCV 4.13 i loader OpenCL z tego samego prefiksu MinGW.

`<prefiks-mingw>` to katalog, w którym leżą `bin`, `include` i `lib`. Przy takiej instalacji jest to często katalog `mingw64` z MSYS2, w innym miejscu niż na maszynie, na której powstały pliki `.vscode`.

## Budowanie

W katalogu projektu, z `<prefiks-mingw>/bin` w `PATH`:

```
cmake -G "MinGW Makefiles" -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DOpenCV_DIR="<prefiks-mingw>/lib/cmake/opencv4" -DOpenCL_INCLUDE_DIR="<prefiks-mingw>/include" -DOpenCL_LIBRARY="<prefiks-mingw>/lib/libOpenCL.dll.a"
cmake --build build
```

To samo polecenie konfiguracji odpala zadanie VS Code `build: harris_app`, tylko gdy brakuje `build/CMakeCache.txt` albo `build/compile_commands.json`. Potem zawsze wykonuje `cmake --build build`.

Katalog `build/` to tylko lokalny wynik CMake i nie wchodzi do repozytorium. Po `cmake --build build` program wykonywalny leży w tym katalogu.

Znaczenie zmiennych:

- `OpenCV_DIR` — katalog zawierający `OpenCVConfig.cmake` (OpenCV z tego prefiksu: `<prefiks-mingw>/lib/cmake/opencv4`)
- `OpenCL_INCLUDE_DIR` — katalog zawierający podkatalog `CL` (z plikiem `cl.h`); tutaj `<prefiks-mingw>/include`
- `OpenCL_LIBRARY` — biblioteka importu OpenCL używana przez ten kompilator (dla MinGW: `<prefiks-mingw>/lib/libOpenCL.dll.a`)

`CMAKE_EXPORT_COMPILE_COMMANDS=ON` zapisuje `build/compile_commands.json` dla IntelliSense.

Przy uruchamianiu katalog `<prefiks-mingw>/bin` musi być w `PATH`, żeby znalazły się DLL-e OpenCV i OpenCL.

## Visual Studio Code

Otwórz w VS Code folder projektu (katalog z `CMakeLists.txt` i `.vscode`, nie sam `build`).

Zainstaluj zalecane rozszerzenia: C/C++ (`ms-vscode.cpptools`) i OpenCL (`galarius.vscode-opencl`). VS Code proponuje je po otwarciu folderu.

Pliki `.vscode` na tej maszynie mają wpisany prefiks `C:/msys64/mingw64`. W `PATH` ten sam katalog jest zapisany jako `C:\msys64\mingw64`. Jeśli kompilator leży gdzie indziej, zamień ten prefiks na swój `<prefiks-mingw>` w:

- `.vscode/tasks.json` (PATH oraz `OpenCV_DIR`, `OpenCL_INCLUDE_DIR`, `OpenCL_LIBRARY`)
- `.vscode/launch.json` (PATH oraz `miDebuggerPath`, czyli `<prefiks-mingw>/bin/gdb.exe`)
- `.vscode/c_cpp_properties.json` (`compilerPath` i ścieżki include)
- `.vscode/settings.json` (cmake, PATH, kompilator i ścieżki include)

Układ względem prefiksu zostaw taki, jak w tych plikach: `bin`, `include`, `include/opencv4`, `lib/cmake/opencv4`, `lib/libOpenCL.dll.a`.

Przycisk Play albo F5 uruchamia konfigurację `harris_app`. Przed startem odpala zadanie `build: harris_app` (`preLaunchTask`), a potem `build/harris_app.exe`. Katalog roboczy (`cwd`) to `build`, więc program znajduje `kernels/harris_gpu.cl`. `PATH` w tej konfiguracji ma zawierać `<prefiks-mingw>/bin`.

`launch.json` zostawia `args` puste. Ścieżka obrazu i próg są w `src/main.cpp`, względem katalogu `build`:

- `kImagePath` — na przykład `../leaf_02/l2nr001.tif`
- `kThreshold` — na przykład `0.02`

Po zmianie tych stałych Play przebudowuje program i uruchamia go od nowa.

## Uruchomienie

CMake kopiuje `kernels/` do katalogu budowania. Program otwiera `kernels/harris_gpu.cl` względem bieżącego katalogu, więc odpalaj go z `build` (tak ustawia `cwd` konfiguracja `harris_app`):

```
cd build
harris_app.exe
```

Obraz i próg biorą się ze stałych `kImagePath` i `kThreshold` w `src/main.cpp`. Współczynnik k wynosi `0.05`. Próg nie jest wyliczany z obrazu; adaptacyjny dobór progu nie jest zaimplementowany.

Na starcie wypisywane są wszystkie platformy OpenCL i ich urządzenia. Jeśli jest karta NVIDIA, program wybiera ją. W przeciwnym razie bierze inną kartę GPU, a gdy GPU nie ma — dowolne urządzenie.

Podgląd zapisuje jako `wyniki/<nazwa>_harris.jpg` w bieżącym katalogu, osobno dla każdego pliku wejściowego. Na konsoli widać użyty próg, liczbę wierzchołków i czasy kerneli.
