CC = g++ -std=c++11
FLAGS = `pkg-config --cflags --libs opencv` -Wall -Wextra

main: camBg.cpp GMM.o
	$(CC) camBg.cpp -o exec GMM.o $(FLAGS)
	rm GMM.o

GMM.o: GMM.cpp GMM.hpp
	$(CC) GMM.cpp -c $(FLAGS)

clean:
	rm exec GMM.o
