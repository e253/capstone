.PHONY: runktest fmt clean

ktest%.exe: kernel_test.cpp common.o v%.o
	g++ -march=native -g -DKERNEL_VER=$* kernel_test.cpp common.o v$*.o -o $@

egtest%.exe: egemv_test.cpp common.o v%.o
	g++ -march=native -g -DVER=$* egemv_test.cpp common.o v$*.o -o $@

bench: bench10.exe bench9.exe bench8.exe bench7.exe
run_bench: bench
	./bench10.exe && sleep 5 && ./bench9.exe && sleep 5 && ./bench8.exe && sleep 5 && ./bench7.exe

bench%.exe: egemv_bench.cpp common.o v%.o
	g++ -march=native -O3 -DVER=$* egemv_bench.cpp common.o v$*.o -o $@ -lpthread

v%.o: v%.cpp
	g++ -march=native -c $< -o $@

common.o: common.cpp
	g++ common.cpp -c -o common.o

fmt:
	clang-format -style=WebKit -i *.cpp *.hpp

clean:
	rm -f *.o *.exe