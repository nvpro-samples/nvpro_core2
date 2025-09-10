/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <implot/implot.h>
#include <imgui/imgui_internal.h>

#include "nvutils/profiler.hpp"

#include "application.hpp"

namespace nvapp {

class ElementProfiler : public IAppElement
{
public:
  typedef enum
  {
    TABLE,
    BAR_CHART,
    PIE_CHART,
    LINE_CHART
  } TabId;

  struct ViewSettings
  {
    const std::string name       = "Profiler";  // name of the view window (must be unique)
    bool              show       = true;        // toggle display of the view window
    TabId             defaultTab = TABLE;       // ID of the tab to open by default
    int               plotHeight = 250;         // height common to all plots
    struct
    {
      bool     detailed = false;  // draw detailed timers avg, min, max, last
      uint32_t levels   = ~0u;    // number of levels to open first
    } table;                      // table settings
    struct
    {
      ImPlotBarGroupsFlags stacked = false;  // draw timers as stacked
    } barChart;                              // barChart settings
    struct
    {
      bool cpuTotal = true;  // Full pie is CPU total time; if false, uses GPU total time
      int  levels   = 1;     // number of levels to draw; 1 = only the root node
    } pieChart;              // pieChart settings
    struct
    {
      bool cpuLine  = true;  // draw higher level CPU timer
      bool gpuLines = true;  // draw GPU timers as lines
      bool gpuFills = true;  // draw GPU timers as filled areas
    } lineChart;             // lineChart settings
  };

private:
  // internal per view storage, hidden from the API
  struct View
  {
    float maxY             = 0.0f;  // max Y axis size for lineChart
    bool  selectDefaultTab = true;  // used to select the default tab at first draw

    std::shared_ptr<ViewSettings> state;  // settings are used as view state
  };

public:
  // defaultViewSettings are optional, but can be used to set different defaults and also to expose
  // to sample code (like hiding views for benchmark through parameter change)
  // some default settings are created internally if not provided
  ElementProfiler(nvutils::ProfilerManager* profiler, std::shared_ptr<ViewSettings> defaultViewSettings = nullptr);

  ~ElementProfiler() = default;

  // add a new view, view name in the state parameter must be unique
  void addView(std::shared_ptr<ViewSettings> state);

  void onAttach(Application* app) override;

  // void onDetach() override {}

  void onUIMenu() override;

  void onUIRender() override;

  // void onRender(VkCommandBuffer /*cmd*/) override {}

private:
  struct EntryNode
  {
    std::string            name;
    float                  cpuTime = 0.f;
    float                  gpuTime = -1.f;
    std::vector<EntryNode> child{};

    nvutils::ProfilerTimeline::TimerInfo timerInfo;

    size_t timerIndex = 0;
  };

  // TODOC
  void updateData(void);

  // TODOC
  uint32_t addEntries(const nvutils::ProfilerTimeline::Snapshot& snapshot,
                      std::vector<EntryNode>&                    nodes,
                      uint32_t                                   startIndex,
                      uint32_t                                   endIndex,
                      uint32_t                                   currentLevel = 0);

  void displayTableNode(const EntryNode& node, bool detailed, uint32_t defaultOpenLevels, uint32_t depth);

  void renderTable(View& view);

  // Rendering the data as a PieChart, showing the percentage of utilization
  void renderPieChart(View& view);

  // Renders the pie chart for a node and up to `numLevels - 1` descendants,
  // where the outer ring has a radius of `plotRadius`.
  // The wedge for the node starts at an angle of `angle0`.
  // 360 degrees == `totalTime`.
  void renderPieChartNode(const EntryNode& node, int level, int numLevels, double plotRadius, double angle0, double totalTime);

  // Rendering the data as a BarChart
  void renderBarChart(View& view);

  // Rendering the data as a cumulated line chart
  void renderLineChart(View& view);

  // Save/read to/from the .ini file to remember the state of the view windows [open/close]
  void addSettingsHandler();

  // draw v-sync toggle
  void drawVsyncCheckbox(void);

  Application*              m_app{nullptr};
  nvutils::ProfilerManager* m_profiler = nullptr;
  std::vector<View>         m_views;
  std::vector<EntryNode>    m_frameNodes;
  std::vector<EntryNode>    m_singleNodes;

  std::vector<nvutils::ProfilerTimeline::Snapshot> m_frameSnapshots;
  std::vector<nvutils::ProfilerTimeline::Snapshot> m_singleSnapshots;
};

}  // namespace nvapp
