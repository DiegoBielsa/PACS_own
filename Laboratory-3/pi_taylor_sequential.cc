#include <iomanip>
#include <iostream>
#include <limits>
#include <chrono>

// Allow to change the floating point type
using my_float = long double;

my_float pi_taylor(size_t steps) {
    int sign = 1;
    float sum = 0;

    for (size_t n = 0; n < steps; n++) {
        sum += sign / static_cast<float>(2 * n + 1);
        sign = -sign;
    }

    return 4.0f * sum;
}

int main(int argc, const char *argv[]) {

    // read the number of steps from the command line
    if (argc != 2) {
        std::cerr << "Invalid syntax: pi_taylor <steps>" << std::endl;
        exit(1);

    }

    // Using time point and system_clock
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    size_t steps = std::stoll(argv[1]);
    auto pi = pi_taylor(steps);
    
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<my_float>::digits10 + 1)
        << pi << std::endl;

    std::cout << "time in seconds: " << elapsed_seconds.count() << "s" << std::endl;
}
