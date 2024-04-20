all: v7 v8

ktest: kernel_test.cpp common.o v7.o v8.o v9.o v10.o
	g++ -march=native -g -DKERNEL_VER=7 kernel_test.cpp common.o v7.o -o ktest7.exe && \
	g++ -march=native -g -DKERNEL_VER=8 kernel_test.cpp common.o v8.o -o ktest8.exe && \
	g++ -march=native -g -DKERNEL_VER=9 kernel_test.cpp common.o v9.o -o ktest9.exe && \
	g++ -march=native -g -DKERNEL_VER=10 kernel_test.cpp common.o v10.o -o ktest10.exe -lpthread && \
	./ktest7.exe && ./ktest8.exe && ./ktest9.exe && ./ktest10.exe

egtest: egemv_test.cpp common.o v7.o v8.o v8_omp.o v9.o v10.o
	g++ -march=native -g -DVER=7 egemv_test.cpp common.o v7.o -o egtest7.exe && \
	g++ -march=native -g -DVER=8 egemv_test.cpp common.o v8.o -o egtest8.exe && \
	g++ -march=native -g -DVER=8.1 egemv_test.cpp common.o v8_omp.o -o egtest8omp.exe -fopenmp && \
	g++ -march=native -g -DVER=9 egemv_test.cpp common.o v9.o -o egtest9.exe && \
	g++ -march=native -g -DVER=10 egemv_test.cpp common.o v10.o -o egtest10.exe && \
	./egtest7.exe && ./egtest8.exe && ./egtest9.exe && ./egtest10.exe && ./egtest8omp.exe

bench: egemv_bench.cpp common.o v7.o v8.o v8_omp.o v9.o v10.o
	g++ -march=native -O3 -DVER=7 egemv_bench.cpp common.o v7.o -o bench7.exe && \
	g++ -march=native -O3 -DVER=8 egemv_bench.cpp common.o v8.o -o bench8.exe && \
	g++ -march=native -O3 -DVER=8.1 egemv_bench.cpp common.o v8_omp.o -o bench8omp.exe -fopenmp && \
	g++ -march=native -O3 -DVER=9 egemv_bench.cpp common.o v9.o -o bench9.exe && \
	g++ -march=native -O3 -DVER=10 egemv_bench.cpp common.o v10.o -o bench10.exe && \
	./bench7.exe && ./bench8.exe && ./bench9.exe && ./bench10.exe && ./bench8omp.exe

v7.o: v7.cpp
	g++ -march=native -c v7.cpp 

v8.o: v8.cpp
	g++ -march=native -c v8.cpp

v8_omp.o: v8_omp.cpp
	g++ -march=native -c v8_omp.cpp

v9.o: v9.cpp
	g++ -march=native -c v9.cpp

v10.o: v10.cpp
	g++ -march=native -c v10.cpp

common: common.cpp
	g++ common.cpp -c -o common.o

fmt:
	clang-format -style=WebKit -i *.cpp *.hpp

clean:
	rm -f *.o *.exe