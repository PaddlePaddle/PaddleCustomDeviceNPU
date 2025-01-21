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

#include <tops/tops_ext.h>

#include "common/utils.h"
#include "gcu/hlir_builder/hlir_builder.h"
#include "paddle/phi/common/data_type.h"

using GcuOp = ::builder::Op;
using GcuOpPtr = std::shared_ptr<GcuOp>;
using GcuPrimitiveType = builder::PrimitiveType;
using GcuType = builder::Type;
// using GcuShape = std::vector<int64_t>;
using GcuBuilder = builder::Builder;
using GcuBuilderPtr = std::shared_ptr<builder::Builder>;
using GcuGraphPtr = std::shared_ptr<hlir::Module>;
// using GcuOpDescPtr = std::shared_ptr<backend::GcuOpDesc>;

namespace custom_engine {
GcuPrimitiveType ConvertFromPhiDataType(const phi::DataType& type);

std::vector<std::string> GetTopsCompileOptions();
topsExecutable_t CompileTopsExecutable(
    const std::shared_ptr<hlir::Module>& module);

}  // namespace custom_engine
