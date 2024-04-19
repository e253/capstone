all: v7 v8

ktest: kernel_test.cpp common.o v7.o v8.o
	g++ -march=native -g -DKERNEL_VER=7 kernel_test.cpp common.o v7.o -o ktest7.exe && \
	g++ -march=native -g -DKERNEL_VER=8 kernel_test.cpp common.o v8.o -o ktest8.exe && \
	./ktest7.exe && ./ktest8.exe

egtest: egemv_test.cpp common.o v7.o v8.o
	g++ -march=native -g -DVER=7 egemv_test.cpp common.o v7.o -o egtest7.exe && \
	g++ -march=native -g -DVER=8 egemv_test.cpp common.o v8.o -o egtest8.exe && \
	./egtest7.exe && ./egtest8.exe

bench: egemv_bench.cpp common.o v7.o v8.o
	g++ -march=native -O3 -DVER=7 egemv_bench.cpp common.o v7.o -o bench7.exe && \
	g++ -march=native -O3 -DVER=8 egemv_bench.cpp common.o v8.o -o bench8.exe && \
	./bench7.exe && ./bench8.exe

v7.o: v7.cpp
	g++ -march=native -c v7.cpp 

v8.o: v8.cpp
	g++ -march=native -c v8.cpp

common: common.cpp
	g++ common.cpp -c -o common.o

fmt:
	clang-format -style=WebKit -i *.cpp *.hpp

clean:
	rm -f *.o *.exe