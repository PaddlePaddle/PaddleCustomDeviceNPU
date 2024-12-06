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

#include "utils/hpu_tracer.h"

HpuTraceParser::HpuTraceParser(uint64_t hpu_start_time)
    : hpu_start_time_ns{hpu_start_time} {}

HpuTraceParser::~HpuTraceParser() {}

void HpuTraceParser::Export(C_Profiler prof,
                            synTraceEvent* events_ptr,
                            size_t num_events,
                            uint64_t wall_start_time) {
  wall_start_time_ns = wall_start_time;
  engine_type_database = EngineDatabase::buildDatabase(events_ptr, num_events);
  initLanes();
  convertEventsToActivities(prof, events_ptr, num_events);
}

bool HpuTraceParser::skipEvent(const synTraceEvent* events_ptr) {
  if (events_ptr->type == 'M') return true;

  auto& engine_types = engine_type_database->engine_types;
  if (engine_types.find(events_ptr->engineType) == engine_types.end())
    return true;

  std::string_view name = events_ptr->name;
  if (name.find("write to mem") != std::string_view::npos) {
    return true;
  }
  return false;
}

bool HpuTraceParser::Contains(const char* haystack, const char* needle) {
  return (haystack != nullptr && needle != nullptr &&
          std::strstr(haystack, needle) != nullptr);
}

void HpuTraceParser::initLanes() {
  auto& engine_types = engine_type_database->engine_types;
  for (auto& e : engine_types) {
    auto& engine_type = e.second;
    auto& engine_type_index = e.first;

    if (!EngineType::isHost(engine_type.name)) {
      for (auto& engine : engine_type.engines) {
        std::string key = std::to_string(engine_type_index) + "_" +
                          std::to_string(engine.index);
        lanes[key] = engine.name;
      }
    }
  }
}

bool HpuTraceParser::isEventInTime(uint64_t start,
                                   uint64_t end,
                                   uint64_t hpu_stop_time_ns) {
  return start > hpu_start_time_ns && end < hpu_stop_time_ns;
}

void HpuTraceParser::convertEventsToActivities(C_Profiler prof,
                                               synTraceEvent* events_ptr,
                                               size_t num_events) {
  struct ActiveEvent {
    synTraceEvent* begin_;
    synTraceEvent* enqueue_;
  };
  using ActiveEventsMap =
      std::unordered_map<uint32_t,
                         std::unordered_map<uint32_t, std::list<ActiveEvent>>>;
  using ActiveEnqueueEventsMap = std::unordered_map<uint32_t, synTraceEvent*>;
  ActiveEventsMap activeEvents;
  ActiveEnqueueEventsMap activeEnqueueEvents;

  for (size_t i{}; i < num_events; i++, events_ptr++) {
    if (skipEvent(events_ptr)) {
      continue;
    }

    if (events_ptr->arguments.recipeId != 0 &&
        Contains(events_ptr->name, "enqueueWithExternalEvents")) {
      activeEnqueueEvents[events_ptr->arguments.recipeId] = events_ptr;
    }

    switch (events_ptr->type) {
      case 'B': {
        auto& eventList =
            activeEvents[events_ptr->engineIndex][events_ptr->contextId];
        ActiveEvent activeEvent{
            events_ptr, activeEnqueueEvents[events_ptr->arguments.recipeId]};
        eventList.push_back(activeEvent);
      } break;
      case 'E': {
#if (CAPTURE_EVENTS & STREAMS_EVENT)
        auto& eventList =
            activeEvents[events_ptr->engineIndex][events_ptr->contextId];
        if (!eventList.empty()) {
          auto begin = timeStampHpuToTB(eventList.front().begin_->timestamp);
          auto end = timeStampHpuToTB(events_ptr->timestamp);
          std::string key = std::to_string(events_ptr->engineType) + "_" +
                            std::to_string(events_ptr->engineIndex);
          std::string engineName = lanes[key];
          std::string name{engineName + ": " + events_ptr->name};
          phi::RuntimeTraceEvent event;

          event.name = name;
          event.start_ns = begin;
          event.end_ns = end;
          event.process_id = events_ptr->engineType;
          event.thread_id = events_ptr->engineIndex;
          event.correlation_id = 0;
          profiler_add_runtime_trace_event(prof, &event);
          eventList.pop_front();
#endif
        }
      } break;
      case 'X': {
#if (CAPTURE_EVENTS & SYNAPSE_API_EVENT)
        auto begin = timeStampHpuToTB(events_ptr->timestamp);
        auto end =
            timeStampHpuToTB(events_ptr->timestamp + events_ptr->duration);
        phi::RuntimeTraceEvent event;
        event.name = events_ptr->name;
        event.start_ns = begin;
        event.end_ns = end;
        event.process_id = static_cast<uint32_t>(getpid());
        event.thread_id = static_cast<uint32_t>(getpid());
        event.correlation_id = 0;
        profiler_add_runtime_trace_event(prof, &event);
#endif
      } break;
    }
  }
}

int64_t HpuTraceParser::timeStampHpuToTB(long double t) {
  uint64_t _t = static_cast<int64_t>(roundl(t * 1000));
  return _t - hpu_start_time_ns + wall_start_time_ns;
}
