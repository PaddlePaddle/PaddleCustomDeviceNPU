// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "graph/graph_executor.h"

namespace custom_graph {

class FeedAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& out = ctx.Output("Out");
    auto col = ctx.Attr<int>("col");
    graph::utils::log() << "[INFO] feed var " << out.Name() << ", col=" << col
                        << std::endl;

    OpCommand("Data").Output(out, "y").Attr("index", col);
    out.MarkAsFeedInput(col);
  }
};

class FetchV2Adapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto col = ctx.Attr<int>("col");
    graph::utils::log() << "[INFO] fetch var " << x.Name() << ", col=" << col
                        << std::endl;

    x.MarkAsFetchOutput(col);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(feed, custom_graph::FeedAdapter);
REG_OP_ADAPTER(fetch_v2, custom_graph::FetchV2Adapter);
