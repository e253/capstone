.PHONY: fmt clean

all:
	clang++ -march=native -O3 cpu.cpp -o cpu.exe

fmt:
	clang-format -style=WebKit -i *.cpp *.hpp

clean:
	rm -f *.o *.exe