// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kolokolova_d_max_of_vector_elements/include/ops_mpi.hpp"

TEST(kolokolova_d_max_of_vector_elements_test, test_pipeline_run) {
  boost::mpi::communicator world;
  int num_processes = world.size();               // Количество запущенных процессов
  int size_rows = num_processes * 2000000;
  std::vector<int> global_max(num_processes, 0);  // Ожидаемый результат
  std::vector<int> global_mat(size_rows);     // Матрица

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int i = 0; i < size_rows; ++i) {
      global_mat[i] = 1;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  
  auto testMpiTaskParallel = std::make_shared<kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel>(taskDataPar, num_processes);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Конвертация в секунды
  };
  // Создание и инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание анализатора производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Печать статистики производительности
  ppc::core::Perf::print_perf_statistic(perfResults);

   if (world.rank() == 0) {
     std::vector<int> results(num_processes);
     std::memcpy(results.data(), taskDataPar->outputs.data(), num_processes * sizeof(int));

    // Ожидаемая проверка результатов
    for (int i = 0; i < num_processes; ++i) {
      EXPECT_EQ(results[i], 1);
    }
   }
}

TEST(kolokolova_d_max_of_vector_elements_test, test_task_run) {
  boost::mpi::communicator world;

  int num_processes = world.size();         // Количество запущенных процессов
  int size_rows = num_processes * 2000000;  // Размер массива

  std::vector<int> global_max(num_processes, INT_MIN);  // Ожидаемый результат
  std::vector<int> global_mat(size_rows);               // Вектор для хранения данных

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  // Заполнение данных с использованием зависимости от ранга
  if (world.rank() == 0) {
    for (int i = 0; i < size_rows; ++i) {
      // Заполняем данные по формуле: (номер строки + 1) * (номер процесса + 1)
      global_mat[i] = (i / num_processes + 1) * (world.rank() + 1);  // Создаем последовательность
    }

    // Установка ожидаемых значений максимума
    for (int i = 0; i < num_processes; ++i) {
      global_max[i] = (2000000 / num_processes) * (i + 1);  // Ожидаемое значение максимума для каждого процесса
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  // Создание экземпляра задачи
  auto testMpiTaskParallel =
      std::make_shared<kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel>(taskDataPar, num_processes);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Создание атрибутов производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Конвертация в секунды
  };

  // Создание и инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание анализатора производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Печать статистики производительности
  ppc::core::Perf::print_perf_statistic(perfResults);

  if (world.rank() == 0) {
    std::vector<int> results(num_processes);
    std::memcpy(results.data(), taskDataPar->outputs.data(), num_processes * sizeof(int));

    // Ожидаемая проверка результатов
    for (int i = 0; i < num_processes; ++i) {
      EXPECT_EQ(results[i], global_max[i]);
    }
  }
}

//TEST(kolokolova_d_max_of_vector_elements_test, test_task_run) {
//  boost::mpi::communicator world;
//  std::vector<int> global_vec;
//  std::vector<int32_t> global_sum(1, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//  int count_size_vector;
//  if (world.rank() == 0) {
//    count_size_vector = 120;
//    global_vec = std::vector<int>(count_size_vector, 1);
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataPar->inputs_count.emplace_back(global_vec.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
//    taskDataPar->outputs_count.emplace_back(global_sum.size());
//  }
//
//  auto testMpiTaskParallel = 
//    std::make_shared<kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel>(taskDataPar, "+");
//  ASSERT_EQ(testMpiTaskParallel->validation(), true);
//  testMpiTaskParallel->pre_processing();
//  testMpiTaskParallel->run();
//  testMpiTaskParallel->post_processing();
//
//  // Create Perf attributes
//  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//  perfAttr->num_running = 10;
//  const boost::mpi::timer current_timer;
//  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
//
//  // Create and init perf results
//  auto perfResults = std::make_shared<ppc::core::PerfResults>();
//
//  // Create Perf analyzer
//  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//  perfAnalyzer->task_run(perfAttr, perfResults);
//  if (world.rank() == 0) {
//    ppc::core::Perf::print_perf_statistic(perfResults);
//    ASSERT_EQ(count_size_vector, global_sum[0]);
//  }
//}

