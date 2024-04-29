.PHONY: all fmt clean

all: cpu igpu

cpu: cpu.cpp
	clang++ -march=native -O3 cpu.cpp -o cpu.exe

igpu: igpu.cpp
	clang++ -march=native -O3 igpu.cpp -o igpu.exe -lOpenCL

ggml: ggml.cpp
	clang++ -march=native -O3 ggml.cpp -o ggml.exe -lggml

fmt:
	clang-format -style=WebKit -i *.cpp *.hpp

clean:
	rm -f *.o *.exe