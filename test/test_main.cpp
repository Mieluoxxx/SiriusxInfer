#include <glog/logging.h>
#include <gtest/gtest.h>

#include <filesystem>

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging("SiriusX");
  // 日志目录路径
  std::string log_dir = "./log/";

  // 检查日志目录是否存在，如果不存在则创建
  if (!std::filesystem::exists(log_dir)) {
    std::filesystem::create_directory(log_dir);
  }

  // 设置日志目录
  FLAGS_log_dir = log_dir;
  FLAGS_alsologtostderr = true;

  LOG(INFO) << "Start Test...\n";
  return RUN_ALL_TESTS();
}