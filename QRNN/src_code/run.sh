g++ qrnn_dict_decode.cpp -o decode -L./ -lopenblas -pthread

ulimit -c unlimited
./decode <input.demo
