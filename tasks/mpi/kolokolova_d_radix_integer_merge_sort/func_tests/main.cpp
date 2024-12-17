#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <random>
#include <vector>

#include "mpi/kolokolova_d_radix_integer_merge_sort/include/ops_mpi.hpp"

using namespace kolokolova_d_radix_integer_merge_sort_mpi;

std::vector<int> kolokolova_d_radix_integer_merge_sort_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  std::uniform_int_distribution<int> dist(-100, 100);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

TEST(kolokolova_d_radix_integer_merge_sort_mpi, Test_Parallel_Sort1) {
  boost::mpi::communicator world;
  int size_vector = world.size() * 4;
  std::vector<int> unsorted_vector(size_vector);
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result(size_vector);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    unsorted_vector = kolokolova_d_radix_integer_merge_sort_mpi::getRandomVector(size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
    taskDataPar->inputs_count.emplace_back(unsorted_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
    taskDataPar->outputs_count.emplace_back(sorted_vector.size());
  }

  kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(size_vector);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
    taskDataSeq->inputs_count.emplace_back(unsorted_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < size_vector; i++) {
      ASSERT_EQ(sorted_vector[i], reference_res[i]);
    }
  }
}

TEST(kolokolova_d_radix_integer_merge_sort_mpi, Test_Parallel_Sort2) {
  boost::mpi::communicator world;
  int size_vector = world.size() * 10;
  std::vector<int> unsorted_vector(size_vector);
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result(size_vector);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    unsorted_vector = kolokolova_d_radix_integer_merge_sort_mpi::getRandomVector(size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
    taskDataPar->inputs_count.emplace_back(unsorted_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
    taskDataPar->outputs_count.emplace_back(sorted_vector.size());
  }

  kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(size_vector);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
    taskDataSeq->inputs_count.emplace_back(unsorted_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < size_vector; i++) {
      ASSERT_EQ(sorted_vector[i], reference_res[i]);
    }
  }
}

TEST(kolokolova_d_radix_integer_merge_sort_mpi, Test_Parallel_Sort3) {
  boost::mpi::communicator world;
  int size_vector = 100;
  std::vector<int> unsorted_vector(size_vector);
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result(size_vector);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    unsorted_vector = kolokolova_d_radix_integer_merge_sort_mpi::getRandomVector(size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
    taskDataPar->inputs_count.emplace_back(unsorted_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
    taskDataPar->outputs_count.emplace_back(sorted_vector.size());
  }

  kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(size_vector);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
    taskDataSeq->inputs_count.emplace_back(unsorted_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < size_vector; i++) {
      ASSERT_EQ(sorted_vector[i], reference_res[i]);
    }
  }
}

TEST(kolokolova_d_radix_integer_merge_sort_mpi, Test_Parallel_Sort_Empty_Input_Vector) {
  boost::mpi::communicator world;
  int size_vector = world.size() * 10;
  std::vector<int> unsorted_vector;
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result(size_vector);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
    taskDataPar->inputs_count.emplace_back(unsorted_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
    taskDataPar->outputs_count.emplace_back(sorted_vector.size());
  }

  if (world.rank() == 0) {
    kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}