.PHONY: fmt clean bench run_bench

ktest%.exe: kernel_test.cpp common.cpp v%.cpp
	g++ -march=native -g -DKERNEL_VER=$* kernel_test.cpp common.cpp v$*.cpp -o $@

egtest%.exe: egemv_test.cpp common.cpp v%.cpp
	g++ -march=native -g -DVER=$* egemv_test.cpp common.cpp v$*.cpp -o $@

bench: bench11.exe bench10.exe bench9.exe bench8.exe bench7.exe
run_bench: bench
	./bench11.exe && sleep 5 && \
	./bench10.exe && sleep 5 && \
	./bench9.exe && sleep 5 &&  \
	./bench8.exe && sleep 5 && \
	./bench7.exe

bench%.exe: egemv_bench.cpp common.cpp v%.cpp
	g++ -march=native -O3 -DVER=$* egemv_bench.cpp common.cpp v$*.cpp -o $@ -lpthread

ibench%.exe: egemv_bench.cpp common.cpp v%.cpp
	icpx -march=native -O3 -DVER=$* egemv_bench.cpp common.cpp v$*.cpp -o $@ -lpthread

fmt:
	clang-format -style=WebKit -i *.cpp *.hpp

clean:
	rm -f *.o *.exe