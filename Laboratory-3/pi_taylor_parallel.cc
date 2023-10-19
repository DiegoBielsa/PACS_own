#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using my_float = long double;

void
pi_taylor_chunk(std::vector<my_float> &output,
        size_t thread_id, size_t start_step, size_t stop_step) {

    int sign = start_step & 0x1 ? -1 : 1;
    for (size_t n = start_step; n < stop_step; n++) {
        output[n] = sign / static_cast<float>(2 * n + 1);
        sign = -sign;
    }
}

std::pair<size_t, size_t>
usage(int argc, const char *argv[]) {
    // read the number of steps from the command line
    if (argc != 3) {
        std::cerr << "Invalid syntax: pi_taylor <steps> <threads>" << std::endl;
        exit(1);
    }

    size_t steps = std::stoll(argv[1]);
    size_t threads = std::stoll(argv[2]);

    if (steps < threads ){
        std::cerr << "The number of steps should be larger than the number of threads" << std::endl;
        exit(1);

    }
    return std::make_pair(steps, threads);
}

int main(int argc, const char *argv[]) {


    auto ret_pair = usage(argc, argv);
    auto steps = ret_pair.first;
    auto threads = ret_pair.second;

    int num_divisions = steps / threads;

    std::vector<my_float> output(steps);
    std::thread thread_pool[threads];

    for (int i = 0; i < threads; i++) {
        int start = i * num_divisions;
        int stop = i == threads - 1 ? steps - 1 : ((i + 1) * num_divisions) - 1;
        thread_pool[i] = std::thread(pi_taylor_chunk, ref(output), i, start, stop);
    }
    
    my_float pi = 0.0f;

    for (size_t i = 0; i < threads; ++i) {
        thread_pool[i].join();
    }

    for (int i = 0; i < steps; i++) {
        pi += output[i];
    }

    pi *= 4;

    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
        << pi << std::endl;
}

