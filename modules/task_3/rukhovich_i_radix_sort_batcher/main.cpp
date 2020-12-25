// Copyright 2020 Igor Rukhovich
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <ctime>
#include <algorithm>
#include <vector>
#include "./radix_sort_batcher.h"

template <class T>
void check_arrays(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        ASSERT_EQ(lhs[i], rhs[i]);
    }
}

void check_par_radix_batcher(size_t num_elements, size_t num_iterations = 10u) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (size_t i = 0u; i < num_iterations; ++i) {
        std::vector<double> array, array_cpy;
        if (rank == 0) {
            array = random_double_array(num_elements);
            array_cpy = array;
        }

        par_radix_sort_batcher(&array);

        if (rank == 0) {
            std::sort(array_cpy.begin(), array_cpy.end());
            check_arrays(array, array_cpy);
        }
    }
}

TEST(RadixSortBatcherMerge, test_radix_empty) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::vector<double> array(0u);
        std::vector<double> array_cpy(array);

        radix_sort(array.begin(), array.end());
        std::sort(array_cpy.begin(), array_cpy.end());

        check_arrays(array, array_cpy);
    }
}

TEST(RadixSortBatcherMerge, test_radix_one_element) {
    std::vector<double> array = random_double_array(1u);
    std::vector<double> array_cpy(array);

    radix_sort(array.begin(), array.end());
    std::sort(array_cpy.begin(), array_cpy.end());

    check_arrays(array, array_cpy);
}

TEST(RadixSortBatcherMerge, test_radix_few_elements) {
    std::vector<double> array = random_double_array(10u);
    std::vector<double> array_cpy(array);

    radix_sort(array.begin(), array.end());
    std::sort(array_cpy.begin(), array_cpy.end());

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    check_arrays(array, array_cpy);
}

TEST(RadixSortBatcherMerge, test_radix_lots_of_elements) {
    std::vector<double> array = random_double_array(10000u);
    std::vector<double> array_cpy(array);

    radix_sort(array.begin(), array.end());
    std::sort(array_cpy.begin(), array_cpy.end());

    check_arrays(array, array_cpy);
}

TEST(RadixSortBatcherMerge, test_batcher_empty) {
    check_par_radix_batcher(0u, 1u);
}

TEST(RadixSortBatcherMerge, test_batcher_one_element) {
    check_par_radix_batcher(1u);
}

TEST(RadixSortBatcherMerge, test_batcher_few_elements) {
    check_par_radix_batcher(10u);
}

TEST(RadixSortBatcherMerge, test_batcher_lots_of_elements) {
    check_par_radix_batcher(10000u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}

/*
double get_time() {
  clock_t time = std::clock();
  return static_cast<double>(time) / CLOCKS_PER_SEC;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> cpy;
    double t, t0, t1, t2;
    if (rank == 0) {
        uint64_t num_elements = 16777216u;
        std::vector<double> arr = random_double_array(num_elements);

        // STD SORT
        cpy = arr;
        t = get_time();
        std::sort(cpy.begin(), cpy.end());
        t0 = get_time() - t;
        std::cout << "std sort done in " << t0 << " seconds\n";

        // RADIX SORT
        cpy = arr;
        t = get_time();
        radix_sort(cpy.begin(), cpy.end());
        t1 = get_time() - t;
        std::cout << "radix sort done in " << t1 << " seconds\n";

        // BATCHER SORT
        cpy = arr;
        t = get_time();
        std::cout << "batcher sort is starting\n";
    }

    par_radix_sort_batcher(&cpy);

    if (rank == 0) {
        t2 = get_time() - t;
        std::cout << "batcher sort done in " << t2 << " seconds\n";
        std::cout << "std, radix, batcher\n";
        std::cout << t0 << ", " << t1 << ", " << t2 << '\n';
    }
    MPI_Finalize();

    return 0;
}
*/
