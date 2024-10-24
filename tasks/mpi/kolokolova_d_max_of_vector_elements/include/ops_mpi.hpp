// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kolokolova_d_max_of_vector_elements_mpi {

std::vector<std::vector<int>> getRandomVector(int sz);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> res;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, int rows_)
      : Task(std::move(taskData_)), rows(std::move(rows_)){}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;  // ���������� ������ ������� ������
  int rows;                      // ���������� �����
  std::vector<int> res;          // ����������
  boost::mpi::communicator world;
  std::vector<int> local_input_;
  int max = 0;
  std::vector<int> elements;
};

}  // namespace kolokolova_d_max_of_vector_elements_mpi