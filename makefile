echo "Compiling main 4.5"
g++ -ggdb main.cpp -o output `pkg-config --cflags --libs opencv4` -latomic && echo "Done"
