// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "custom_engine/ir_translator/utils/utils.h"

#include "gcu/tops_graph_compiler/tops_graph_compiler.h"
#include "gcu/tops_graph_compiler/tops_graph_compiler_option.h"
#include "paddle/phi/core/dense_tensor.h"

namespace custom_engine {
namespace {
std::vector<std::string> TargetOptionSplit(const std::string& s,
                                           char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    std::size_t first_non_space = token.find_first_not_of(" \t\n\r");
    std::size_t last_non_space = token.find_last_not_of(" \t\n\r");
    if (first_non_space == std::string::npos ||
        last_non_space == std::string::npos) {
      continue;
    }
    token.substr(first_non_space, last_non_space - first_non_space + 1);
    if (!token.empty()) tokens.push_back(token);
  }
  return tokens;
}
}  // namespace

GcuPrimitiveType ConvertFromPhiDataType(const phi::DataType& type) {
  switch (type) {
    case phi::DataType::BOOL:
      return builder::PrimitiveType::PRED();
    case phi::DataType::INT8:
      return builder::PrimitiveType::S8();
    case phi::DataType::INT16:
      return builder::PrimitiveType::S16();
    case phi::DataType::INT32:
      return builder::PrimitiveType::S32();
    case phi::DataType::INT64:
      return builder::PrimitiveType::S64();
    case phi::DataType::FLOAT16:
      return builder::PrimitiveType::F16();
    case phi::DataType::FLOAT32:
      return builder::PrimitiveType::F32();
    case phi::DataType::FLOAT64:
      return builder::PrimitiveType::F64();
    case phi::DataType::UINT8:
      return builder::PrimitiveType::U8();
    case phi::DataType::UINT16:
      return builder::PrimitiveType::U16();
    case phi::DataType::UINT32:
      return builder::PrimitiveType::U32();
    case phi::DataType::UINT64:
      return builder::PrimitiveType::U64();

    default:
      return builder::PrimitiveType::NONE();
  }
}

std::vector<std::string> GetTopsCompileOptions() {
  std::vector<std::string> opts;

  auto target_name = custom_kernel::GetTargetName();
  //   std::string hlir_options = "hlir-codegen-pipeline";
  std::string hlir_options = "tops-hlir-pipeline";

  // add target options
  int options_len = 1024;            // NOLINT
  char target_options[options_len];  // NOLINT
  TOPSGRAPH_CHECK(
      topsgraphInitOptions(target_name.c_str(), target_options, options_len));

  std::string target_opt_s = std::string(target_options);
  char delimiter = '-';
  auto target_opt_vec = TargetOptionSplit(target_opt_s, delimiter);
  for (auto it : target_opt_vec) {
    auto temp_opt = "-" + it;
    opts.emplace_back(temp_opt);
  }
  opts.emplace_back(std::string("-hlir=") + hlir_options);
  //   opts.emplace_back(
  //       std::string("-codegen=codegen-gcu-pipeline{enable-memory-reuse=true}"));
  //   opts.emplace_back(std::string("-output=codegen"));

  if (VLOG_IS_ON(3)) {
    std::stringstream ss;
    ss << "compile options: ";
    for (auto it : opts) {
      ss << it << " ";
    }
    VLOG(3) << ss.str();
  }

  return opts;
}

topsExecutable_t CompileTopsExecutable(
    const std::shared_ptr<hlir::Module>& module) {
  std::vector<const char*> options;
  auto compile_options = GetTopsCompileOptions();
  for (auto& option : compile_options) {
    options.push_back(option.c_str());
  }

  // create program and compile
  topsgraphProgram program;
  TOPSGRAPH_CHECK(topsgraphCreateProgramFromModule(&program, module.get()));
  TOPSGRAPH_CHECK(
      topsgraphCompileProgram(program, options.size(), options.data()));

  // get binary size and binary data
  uint64_t binary_size = 0;
  TOPSGRAPH_CHECK(topsgraphGetBinSize(program, &binary_size));
  std::unique_ptr<char[]> binary(new char[binary_size]);
  TOPSGRAPH_CHECK(topsgraphGetBin(program, binary.get()));

  // delete program
  topsgraphDestroyProgram(&program);

  topsExecutable_t exe;
  RT_CHECK(topsCreateExecutable(&exe, binary.get(), binary_size));

  return exe;
}

}  // namespace custom_engine
