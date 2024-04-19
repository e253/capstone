all: v7 v8

test: test.cpp common.o v7.o
	g++ -march=native -g test.cpp common.o v7.o -o test

v7: v7.cpp
	g++ -march=native -g v7.cpp -c -o v7.o

v8: v8.cpp
	g++ -march=native -O3 v8.cpp -o v8

common: common.cpp
	g++ common.cpp -c -o common.o

fmt:
	clang-format -style=WebKit -i *.cpp *.hpp

clean:
	rm -f *.o v7 v8 test *.exe