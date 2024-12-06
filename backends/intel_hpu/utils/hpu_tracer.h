// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
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

#include <unistd.h>

#include <cmath>
#include <cstring>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "habanalabs/synapse_api.h"
#include "paddle/phi/api/profiler/trace_event.h"
#include "paddle/phi/backends/device_ext.h"

#define STREAMS_EVENT 0x1
#define SYNAPSE_API_EVENT 0x2
#define CAPTURE_EVENTS (STREAMS_EVENT | SYNAPSE_API_EVENT)

struct EngineType {
  struct Engine {
    uint32_t index;
    std::string name;
  };
  std::string name;
  std::vector<Engine> engines;

  static bool isInteresting(const std::string_view name) {
    return std::strstr(name.data(), "*TPC") ||
           std::strstr(name.data(), "*MME") ||
           std::strstr(name.data(), "*EDMA") ||
           std::strstr(name.data(), "*PDMA") ||
           std::strstr(name.data(), "*KDMA");
  }
  static bool isHost(const std::string_view name) {
    static std::string host_meta_name = "***Host";
    return name == host_meta_name;
  }
};

struct EngineDatabase {
  EngineDatabase() {}

  std::unordered_map<uint32_t, EngineType> engine_types;

  static std::unique_ptr<EngineDatabase> buildDatabase(
      synTraceEvent* events_ptr, size_t num_events) {
    std::unique_ptr<EngineDatabase> result = std::make_unique<EngineDatabase>();
    auto& engine_types = result->engine_types;

    synTraceEvent* host_meta_event_ptr = events_ptr;
    for (uint64_t i = 0; i < num_events; i++, host_meta_event_ptr++) {
      if (host_meta_event_ptr->type != 'M') break;
      if (host_meta_event_ptr->engineIndex == 0 &&
          EngineType::isHost(host_meta_event_ptr->arguments.name)) {
        auto& engine_type = engine_types[host_meta_event_ptr->engineType];
        engine_type.name = host_meta_event_ptr->arguments.name;
        break;
      }
    }

    for (uint64_t i = 0; i < num_events; i++, events_ptr++) {
      if (events_ptr->type != 'M') {
        continue;
      }

      if (events_ptr->engineIndex == 0 &&
          EngineType::isInteresting(events_ptr->arguments.name)) {
        auto& engine_type = engine_types[events_ptr->engineType];
        engine_type.name = events_ptr->arguments.name;
        continue;
      }
      auto engine_type_it = engine_types.find(events_ptr->engineType);
      if (engine_type_it != engine_types.end()) {
        auto& engine_type = engine_type_it->second;
        engine_type.engines.push_back({.index = events_ptr->engineIndex,
                                       .name = events_ptr->arguments.name});
      }
    }
    return result;
  }
};

class HpuTraceParser {
 public:
  explicit HpuTraceParser(uint64_t hpu_start_time);

  ~HpuTraceParser();

  void Export(C_Profiler prof,
              synTraceEvent* events_ptr,
              size_t num_events,
              uint64_t wall_stop_time);

 private:
  bool skipEvent(const synTraceEvent* events_ptr);
  bool Contains(const char* haystack, const char* needle);

  void initLanes();
  bool isEventInTime(uint64_t start, uint64_t end, uint64_t hpu_stop_time);
  void convertEventsToActivities(C_Profiler prof,
                                 synTraceEvent* events_ptr,
                                 size_t num_events);
  int64_t timeStampHpuToTB(long double t);
  std::unordered_map<std::string, std::string> lanes;
  uint64_t hpu_start_time_ns;
  uint64_t wall_start_time_ns;
  std::unique_ptr<EngineDatabase> engine_type_database;
};
