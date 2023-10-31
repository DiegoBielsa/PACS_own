#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <chrono>

using my_float = long double;

typedef struct {
    size_t large_chunk;
    size_t small_chunk;
    size_t split_item;
} chunk_info;

// For a given number of iterations N and threads
// the iterations are divided:
// N % threads receive N / threads + 1 iterations
// the rest receive N / threads
constexpr chunk_info
split_evenly(size_t N, size_t threads)
{
    return {N / threads + 1, N / threads, N % threads};
}

std::pair<size_t, size_t>
get_chunk_begin_end(const chunk_info& ci, size_t index)
{
    size_t begin = 0, end = 0;
    if (index < ci.split_item ) {
        begin = index*ci.large_chunk;
        end = begin + ci.large_chunk; // (index + 1) * ci.large_chunk
    } else {
        begin = ci.split_item*ci.large_chunk + (index - ci.split_item) * ci.small_chunk;
        end = begin + ci.small_chunk;
    }
    return std::make_pair(begin, end);
}

void
pi_taylor_chunk(std::vector<my_float> &output,
        size_t thread_id, size_t start_step, size_t stop_step) {

    int sign = start_step & 0x1 ? -1 : 1;
    for (size_t n = start_step; n < stop_step; n++) {
        output[thread_id] += sign / static_cast<float>(2 * n + 1);
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

    threads = std::min(uint(threads), std::thread::hardware_concurrency());
    std::cout << "Your machine supports " << std::thread::hardware_concurrency() << " threads. Therefore, the number of threads to launch is " << threads << std::endl;

    // Using time point and system_clock
    std::chrono::time_point<std::chrono::system_clock> global_start, global_end;
    global_start = std::chrono::system_clock::now();

    std::vector<my_float> output(threads, 0.0f);
    std::vector<std::thread> thread_vector;

    auto chunks = split_evenly(steps, threads);
    for(size_t i = 0; i < threads; ++i) {
        auto begin_end = get_chunk_begin_end(chunks, i);
        thread_vector.push_back(std::thread(pi_taylor_chunk, ref(output), i, begin_end.first, begin_end.second));
        //std::cout << i << ", " << begin_end.first << ", " << begin_end.second << std::endl;
    }

    my_float pi = 0.0f;
    // wait for completion
    for(size_t i = 0; i < threads; ++i) {
        thread_vector[i].join();
        pi += output[i];
    }
    pi *= 4.0f;
    
    global_end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = global_end - global_start;
    

    std::cout << "For " << steps << " steps, and " << threads << " threads, pi value: "
        << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
        << pi << std::endl;
    
    std::cout << " TOTAL time in seconds: " << elapsed_seconds.count() << "s" << std::endl;

}

