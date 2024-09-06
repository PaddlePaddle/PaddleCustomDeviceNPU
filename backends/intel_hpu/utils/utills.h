// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <assert.h>

#include <cstdarg>
#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"

template <class KEY_T, class VAL_T>
class LRUCache {
 private:
  std::list<std::pair<KEY_T, VAL_T>> item_list;
  std::unordered_map<KEY_T, decltype(item_list.begin())> item_map;
  size_t cache_size;

 private:
  void clean(void) {
    while (item_map.size() > cache_size) {
      auto last_it = item_list.end();
      last_it--;
      item_map.erase(last_it->first);
      item_list.pop_back();
    }
  }

 public:
  explicit LRUCache(int cache_size_) : cache_size(cache_size_) {}

  void put(const KEY_T& key, const VAL_T& val) {
    auto it = item_map.find(key);
    if (it != item_map.end()) {
      item_list.erase(it->second);
      item_map.erase(it);
    }
    item_list.push_front(make_pair(key, val));
    item_map.insert(make_pair(key, item_list.begin()));
    clean();
  }

  bool exist(const KEY_T& key) { return (item_map.count(key) > 0); }

  VAL_T get(const KEY_T& key) {
    auto it = item_map.find(key);
    if (it != item_map.end()) {
      item_list.splice(item_list.begin(), item_list, it->second);
      return it->second->second;
    } else {
      return nullptr;
    }
  }
};

class KeyCreator {
 public:
  KeyCreator() { key_.reserve(kMaxKeyLength); }

  ~KeyCreator() {}

  void AddAsKey(const std::string& str) {
    key_.append(str);
    key_.append(1, delimiter);
  }

  template <typename T>
  void AddAsKey(const T data) {
    auto buffer = reinterpret_cast<const char*>(&data);
    key_.append(buffer, sizeof(T));
    key_.append(1, delimiter);
  }

  void AddAsKey(const std::vector<int64_t>& dims) {
    for (unsigned int i = 0; i < dims.size(); i++) {
      AddAsKey<int>(dims[i]);
    }
  }

  std::string GetKey() { return key_; }

 private:
  std::string key_ = "";
  const char delimiter = '_';
  const int kMaxKeyLength = 256;
};

class OpCacheOperator {
 public:
  template <typename T, typename TP>
  void prepareOpInfo(std::string guid_prefix,
                     const std::vector<DIMS>& ins,
                     TP* params) {
    if (std::is_same<T, phi::dtype::float16>::value) {
      guid_ = guid_prefix + "_f16";
      datatype_ = syn_type_fp16;
    } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
      guid_ = guid_prefix + "_bf16";
      datatype_ = syn_type_bf16;
    } else if (std::is_same<T, float>::value) {
      guid_ = guid_prefix + "_f32";
      datatype_ = syn_type_single;
    } else if (std::is_same<T, phi::dtype::float8_e4m3fn>::value) {
      datatype_ = syn_type_fp8_143;
      guid_ = guid_prefix + "_hf8";
    } else if (std::is_same<T, int32_t>::value) {
      datatype_ = syn_type_int32;
      guid_ = guid_prefix + "_i32";
    } else if (std::is_same<T, bool>::value) {
      datatype_ = syn_type_int8;
      guid_ = guid_prefix + "_i8";
    } else if (std::is_same<T, int64_t>::value) {
      datatype_ = syn_type_int64;
      guid_ = guid_prefix + "_i64";
    } else {
      PD_CHECK(
          false, "[RUNTIME] synDataType not supported = %s", typeid(T).name());
    }

    key_creator_.AddAsKey(guid_);
    for (std::size_t i = 0; i < ins.size(); i++) {
      key_creator_.AddAsKey(ins[i]);
    }
    if (params != nullptr) key_creator_.AddAsKey<TP>(*params);
  }

  synRecipeHandle GetRecipe() {
    auto& lru_cache = OpCacheOperator::GetLRUCache();
    if (lru_cache.exist(key_creator_.GetKey())) {
      return lru_cache.get(key_creator_.GetKey());
    } else {
      return nullptr;
    }
  }

  void setOp(HpuOperator op) {
    auto& lru_cache = OpCacheOperator::GetLRUCache();
    lru_cache.put(key_creator_.GetKey(), op.GetRecipe());
    return;
  }

  template <typename T>
  std::string GetTypeStr() {
    if (std::is_same<T, phi::dtype::float16>::value) {
      return "f16";
    } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
      return "bf16";
    } else if (std::is_same<T, float>::value) {
      return "f32";
    } else if (std::is_same<T, phi::dtype::float8_e4m3fn>::value) {
      return "hf8";
    } else {
      // PD_CHECK("synDataType not supported");
      return "";
    }
  }

 private:
  static inline LRUCache<std::string, synRecipeHandle>& GetLRUCache() {
    static const int kCapacity = 1024;  // cache capacity
    static LRUCache<std::string, synRecipeHandle> lru_cache_(kCapacity);
    return lru_cache_;
  }

 public:
  std::string guid_;
  synDataType datatype_;
  KeyCreator key_creator_;
};

std::string ShowErrorMsg(synStatus s);
