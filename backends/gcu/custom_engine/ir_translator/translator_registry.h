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

#pragma once

#include <glog/logging.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "custom_engine/ir_translator/utils/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/operation.h"

namespace custom_engine {

using OpTranslateFunc = std::function<GcuOpPtr(
    GcuBuilderPtr builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &map_inputs)>;

static inline std::string OpTranslateFuncKey(const std::string &op_name) {
  std::regex pattern("\\.");
  std::string result = std::regex_replace(op_name, pattern, "_");
  return result;
}

class TranslatorRegistry {
 public:
  static TranslatorRegistry &Instance() {
    static TranslatorRegistry g_op_translator_registry_instance;
    return g_op_translator_registry_instance;
  }

  bool Has(const std::string &op_name) const {
    return translator_map_.find(op_name) != translator_map_.end();
  }

  void Insert(const std::string &op_name,
              const OpTranslateFunc &op_trans_func) {
    PADDLE_ENFORCE_NE(
        Has(op_name),
        true,
        common::errors::InvalidArgument(
            "OpTranslateFunc of %s has been registered.", op_name));
    translator_map_.insert({op_name, op_trans_func});
    std::cout << "TranslatorRegistry insert " << op_name << std::endl;
  }

  OpTranslateFunc Get(const std::string &op_name) const {
    PADDLE_ENFORCE_EQ(
        Has(op_name),
        true,
        common::errors::InvalidArgument(
            "OpTranslateFunc of %s has not been registered.", op_name));
    return translator_map_.at(op_name);
  }

 private:
  TranslatorRegistry() = default;
  std::unordered_map<std::string, OpTranslateFunc> translator_map_;

  TranslatorRegistry(const TranslatorRegistry &) = delete;
  TranslatorRegistry(TranslatorRegistry &&) = delete;
  TranslatorRegistry &operator=(const TranslatorRegistry &) = delete;
  TranslatorRegistry &operator=(TranslatorRegistry &&) = delete;
};

class OpTranslatorRegistrar {
 public:
  // The action of registration is in the constructor of a global registrar
  // variable, which are not used in the code that calls package framework, and
  // would be removed from the generated binary file by the linker. To avoid
  // such removal, we add Touch to all registrar classes and make
  // USE_OP_TRANSLATOR macros to call this method. So, as long as the callee
  // code calls USE_OP_TRANSLATOR, the global registrar variable won't be
  // removed by the linker.
  void Touch() {}
  OpTranslatorRegistrar(const char *op_name,
                        const OpTranslateFunc &op_trans_func) {
    TranslatorRegistry::Instance().Insert(op_name, op_trans_func);
  }
};

#define STATIC_ASSERT_TRANSLATOR_GLOBAL_NAMESPACE(uniq_name, msg)              \
  struct __test_translator_global_namespace_##uniq_name##__ {};                \
  static_assert(                                                               \
      std::is_same<::__test_translator_global_namespace_##uniq_name##__,       \
                   __test_translator_global_namespace_##uniq_name##__>::value, \
      msg)

// Register a new op_trans_func that can be applied on the operator.
#define REGISTER_OP_TRANSLATOR(op_name, op_trans_func)                  \
  STATIC_ASSERT_TRANSLATOR_GLOBAL_NAMESPACE(                            \
      __reg_op_translator__##op_name,                                   \
      "REGISTER_OP_TRANSLATOR must be called in global namespace");     \
  static custom_engine::OpTranslatorRegistrar                           \
      __op_translator_registrar_##op_name##__(#op_name, op_trans_func); \
  int TouchOpTranslatorRegistrar_##op_name() {                          \
    __op_translator_registrar_##op_name##__.Touch();                    \
    return 0;                                                           \
  }

#define USE_OP_TRANSLATOR(op_name)                             \
  STATIC_ASSERT_TRANSLATOR_GLOBAL_NAMESPACE(                   \
      __use_op_translator_itself_##op_name,                    \
      "USE_OP_TRANSLATOR must be called in global namespace"); \
  extern int TouchOpTranslatorRegistrar_##op_name();           \
  static int use_op_translator_itself_##op_name##_ UNUSED =    \
      TouchOpTranslatorRegistrar_##op_name()

}  // namespace custom_engine
