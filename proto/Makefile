library:
	c++ -fPIC --shared fourier_basic.cpp -o fourier_basic.so -std=c++17 -lstdc++ -O3 -mavx

main: library
	echo "Main"