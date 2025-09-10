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

#include <nvgui/fonts.hpp>
#include <nvvk/debug_util.hpp>
#include <nvutils/logger.hpp>

#include "elem_profiler.hpp"

using namespace nvapp;

// defaultViewSettings are optional, but can be used to set different defaults and also to expose
// to sample code (like hiding views for benchmark through parameter change)
// some default settings are created internally if not provided
ElementProfiler::ElementProfiler(nvutils::ProfilerManager* profiler, std::shared_ptr<ViewSettings> defaultViewSettings)
    : m_profiler(profiler)
{
  m_views.push_back({.state = defaultViewSettings ? defaultViewSettings : std::make_shared<nvapp::ElementProfiler::ViewSettings>()});
};

void ElementProfiler::onAttach(Application* app)
{
  m_app = app;
  //
  addSettingsHandler();
}

// add a new view, view name in the state parameter must be unique
void ElementProfiler::addView(std::shared_ptr<ViewSettings> state)
{
  // check if a view with same name already exists
  for(const auto& existingView : m_views)
  {
    if(existingView.state->name == state->name)
    {
      LOGE("Fatal: view named %s already exists", state->name.c_str());
      return;
    }
  }
  // prepare our internal view and add it
  m_views.push_back({.state = std::move(state)});
}

void ElementProfiler::onUIMenu()
{
  if(ImGui::BeginMenu("View"))
  {
    for(auto& view : m_views)
    {
      ImGui::MenuItem((ICON_MS_BLOOD_PRESSURE " " + view.state->name).c_str(), "", &view.state->show);
    }
    ImGui::EndMenu();
  }
}

void ElementProfiler::onUIRender()
{
  constexpr float deltaTime     = (1.0f / 60.0f);  // Frequency 60Hz
  static float    s_timeElapsed = 0;
  s_timeElapsed += ImGui::GetIO().DeltaTime;

  // check if at least one view is visible
  bool showWindow = true;
  for(const auto& view : m_views)
  {
    showWindow &= view.state->show;
  }

  // collecting data if needed
  if(s_timeElapsed >= deltaTime)
  {
    s_timeElapsed = 0;

    updateData();
  }

  // display each visible view
  for(auto& view : m_views)
  {
    if(!view.state->show)
      continue;

    // Opening the window
    if(ImGui::Begin(view.state->name.c_str(), &view.state->show))
    {

      if(ImGui::BeginTabBar("Profiler Tabs"))
      {
        if(ImGui::BeginTabItem("Table", NULL, view.selectDefaultTab && view.state->defaultTab == TABLE ? ImGuiTabItemFlags_SetSelected : 0))
        {
          renderTable(view);
          ImGui::EndTabItem();
        }
        if(ImGui::BeginTabItem("BarChart", NULL,
                               view.selectDefaultTab && view.state->defaultTab == BAR_CHART ? ImGuiTabItemFlags_SetSelected : 0))
        {
          renderBarChart(view);
          ImGui::EndTabItem();
        }
        if(ImGui::BeginTabItem("LineChart", NULL,
                               view.selectDefaultTab && view.state->defaultTab == LINE_CHART ? ImGuiTabItemFlags_SetSelected : 0))
        {
          renderLineChart(view);
          ImGui::EndTabItem();
        }
        if(ImGui::BeginTabItem("PieChart", NULL,
                               view.selectDefaultTab && view.state->defaultTab == PIE_CHART ? ImGuiTabItemFlags_SetSelected : 0))
        {
          renderPieChart(view);
          ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
      }

      view.selectDefaultTab = false;
    }
    ImGui::End();
  }
}

/////////////////
// Private methods

void ElementProfiler::updateData()
{

  // retrieve statistics, thread safe
  m_profiler->getSnapshots(m_frameSnapshots, m_singleSnapshots);

  // reset tree storage
  m_frameNodes.clear();
  m_singleNodes.clear();

  // fill tree
  for(const auto& snapshot : m_frameSnapshots)
  {
    EntryNode entryNode;
    entryNode.name = snapshot.name;
    m_frameNodes.emplace_back(entryNode);
    addEntries(snapshot, m_frameNodes.back().child, 0, (uint32_t)snapshot.timerInfos.size());
  }
  for(const auto& snapshot : m_singleSnapshots)
  {
    EntryNode entryNode;
    entryNode.name = snapshot.name;
    m_singleNodes.emplace_back(entryNode);
    addEntries(snapshot, m_singleNodes.back().child, 0, (uint32_t)snapshot.timerInfos.size());
  }

  // update line plot history if needed
  bool updateHistory = true;
  for(const auto& view : m_views)
  {
    updateHistory &= view.state->show;
  }
}

uint32_t ElementProfiler::addEntries(const nvutils::ProfilerTimeline::Snapshot& snapshot,
                                     std::vector<EntryNode>&                    nodes,
                                     uint32_t                                   startIndex,
                                     uint32_t                                   endIndex,
                                     uint32_t                                   currentLevel)
{
  for(uint32_t curIndex = startIndex; curIndex < endIndex; curIndex++)
  {
    const auto& timerInfo = snapshot.timerInfos[curIndex];
    const auto& level     = timerInfo.level;

    if(level < currentLevel)
      return curIndex;

    const auto& name = snapshot.timerNames[curIndex];

    EntryNode entryNode;
    entryNode.name      = name.empty() ? "N/A" : name;
    entryNode.gpuTime   = static_cast<float>(timerInfo.gpu.average / 1000.);
    entryNode.cpuTime   = static_cast<float>(timerInfo.cpu.average / 1000.);
    entryNode.timerInfo = timerInfo;

    if(timerInfo.async)
    {
      nodes.emplace_back(entryNode);
      continue;
    }

    uint32_t nextLevel = curIndex + 1 < endIndex ? snapshot.timerInfos[curIndex + 1].level : currentLevel;
    if(nextLevel > currentLevel)
    {
      curIndex = addEntries(snapshot, entryNode.child, curIndex + 1, endIndex, nextLevel);
    }
    nodes.emplace_back(entryNode);
    if(nextLevel < currentLevel)
      return curIndex;
  }
  return endIndex;
}

void ElementProfiler::displayTableNode(const EntryNode& node, bool detailed, uint32_t defaultOpenLevels, uint32_t depth)
{
  ImGuiTableFlags flags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_SpanAllColumns;

  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  const bool is_folder = (node.child.empty() == false);
  flags                = is_folder ? flags | (depth < defaultOpenLevels ? ImGuiTreeNodeFlags_DefaultOpen : 0) :
                                     flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  bool        open     = ImGui::TreeNodeEx(node.name.c_str(), flags);
  const auto& info     = node.timerInfo;

  // convert to microseconds and draw otherwise display '--' if invalid
  auto drawValue = [](double value) {
    if(value <= 0)
      ImGui::TextDisabled("--");
    else
      ImGui::Text("%3.3f", value / 1000.0f);
  };

  ImGui::PushFont(nvgui::getMonospaceFont());
  ImGui::TableNextColumn();
  drawValue(info.gpu.average);
  if(detailed)
  {
    ImGui::TableNextColumn();
    drawValue(info.gpu.last);
    ImGui::TableNextColumn();
    drawValue(info.gpu.absMinValue);
    ImGui::TableNextColumn();
    drawValue(info.gpu.absMaxValue);
  }
  ImGui::TableNextColumn();
  drawValue(info.cpu.average);
  if(detailed)
  {
    ImGui::TableNextColumn();
    drawValue(info.cpu.last);
    ImGui::TableNextColumn();
    drawValue(info.cpu.absMinValue);
    ImGui::TableNextColumn();
    drawValue(info.cpu.absMaxValue);
  }
  ImGui::PopFont();

  if((open) && is_folder)
  {
    for(int child_n = 0; child_n < static_cast<int>(node.child.size()); child_n++)
    {
      displayTableNode(node.child[child_n], detailed, defaultOpenLevels, depth + 1);
    }
    ImGui::TreePop();
  }
}

void ElementProfiler::drawVsyncCheckbox()
{
  bool       vsync   = m_app->isVsync();
  const bool showRed = vsync;

  if(showRed)
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));

  bool changed = ImGui::Checkbox("V-Sync", &vsync);

  if(showRed)
    ImGui::PopStyleColor();  // Revert to previous color

  if(ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
    ImGui::SetTooltip("Disable V-Sync to measure nominal performance.");

  if(changed)
    m_app->setVsync(vsync);
}

void ElementProfiler::renderTable(View& view)
{
  bool copy = false;

  drawVsyncCheckbox();

  ImGui::SameLine();
  ImGui::Checkbox("detailed", &view.state->table.detailed);

  // Copy content
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 38);
  if(ImGui::Button(ICON_MS_CONTENT_COPY))
  {
    ImGui::LogToClipboard();
    copy = true;
  }
  if(ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
    ImGui::SetTooltip("Copy data to clipboard");

  const int minGridSize = view.state->table.detailed ? 1500 : 550;  // minimum size of container for responsive grid mode
  const bool  gridMode = ImGui::GetContentRegionAvail().x >= minGridSize && m_frameNodes.size() > 1;
  const float width    = ImGui::GetContentRegionAvail().x / (gridMode ? 2.0f : 1.0f) - 5.0f;

  const ImGuiTableFlags tableFlags = ImGuiTableFlags_BordersV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_Resizable
                                     | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoBordersInBody;

  for(size_t i = 0; i < m_frameNodes.size(); ++i)
  {
    if(i > 0)
    {
      if(gridMode)
        ImGui::SameLine();
      else
        ImGui::Spacing();  // add a small vertical gap between tables, for better legibility
    }

    if(!m_frameNodes[i].child.empty() || m_singleNodes[i].child.empty())
    {
      int colCount = view.state->table.detailed ? 9 : 3;

      if(ImGui::BeginTable("EntryTable", colCount, tableFlags, ImVec2(width, 0)))
      {
        ImGui::TableSetupColumn(m_frameNodes[i].name.c_str(), ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthFixed, 250.);

        ImGui::TableSetupColumn("GPU avg", ImGuiTableColumnFlags_WidthStretch);
        if(view.state->table.detailed)
        {
          ImGui::TableSetupColumn("GPU last", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("GPU min", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("GPU max", ImGuiTableColumnFlags_WidthStretch);
        }
        ImGui::TableSetupColumn("CPU avg", ImGuiTableColumnFlags_WidthStretch);
        if(view.state->table.detailed)
        {
          ImGui::TableSetupColumn("CPU last", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("CPU min", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("CPU max", ImGuiTableColumnFlags_WidthStretch);
        }
        ImGui::TableHeadersRow();

        for(const auto& node : m_frameNodes[i].child)
        {
          displayTableNode(node, view.state->table.detailed, view.state->table.levels, 0);
        }

        for(const auto& node : m_singleNodes[i].child)
        {
          displayTableNode(node, view.state->table.detailed, view.state->table.levels, 0);
        }

        ImGui::EndTable();
      }
    }
  }
  if(copy)
  {
    ImGui::LogFinish();
  }
}


void ElementProfiler::renderPieChart(View& view)
{
  const bool  gridMode = ImGui::GetContentRegionAvail().x >= 600 && m_frameNodes.size() > 1;
  const float width    = ImGui::GetContentRegionAvail().x / (gridMode ? 2.0f : 1.0f) - 5.0f;

  const float legendWidth = 170.0f;  // Estimate the width needed for the legend
  const float chartWidth  = (ImGui::GetContentRegionAvail().x - legendWidth) / (gridMode ? 2.0f : 1.0f) - 5.0f;

  drawVsyncCheckbox();

  ImGui::SameLine();
  ImGui::Checkbox("CPU total", &view.state->pieChart.cpuTotal);
  if(ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
    ImGui::SetTooltip("Use CPU frame time as total time, otherwise use sum of GPU timers.");

  ImGui::SameLine();
  ImGui::SetNextItemWidth(100.F);
  int& levels = view.state->pieChart.levels;
  ImGui::InputInt("Levels", &levels);
  levels = std::max(1, levels);  // Make sure it's always >= 1

  for(auto i = 0; i < m_frameNodes.size(); ++i)
  {
    const auto& rootNode = m_frameNodes[i];

    if(!rootNode.child.empty())
    {
      const auto& node = rootNode.child[0];

      if(gridMode && i % 2 != 0)
        ImGui::SameLine();

      // use different color palette for better legibility
      if(i % 3 == 0)
        ImPlot::PushColormap(ImPlotColormap_Deep);
      if(i % 3 == 1)
        ImPlot::PushColormap(ImPlotColormap_Pastel);
      if(i % 3 == 2)
        ImPlot::PushColormap(ImPlotColormap_Viridis);

      if(ImPlot::BeginPlot(rootNode.name.c_str(), ImVec2(width, (float)view.state->plotHeight), ImPlotFlags_Equal | ImPlotFlags_NoMouseText))
      {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Lock,
                          ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Lock);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);

        const float aspectRatio = 0.5f * (float)chartWidth / (float)view.state->plotHeight;
        ImPlot::SetupAxesLimits(0.5 - aspectRatio, 0.5 + aspectRatio, 0, 1, ImPlotCond_Always);

        const double totalTime = (view.state->pieChart.cpuTotal ? node.cpuTime : node.gpuTime);
        renderPieChartNode(node, 0, levels, 0.4, 90.0, totalTime);

        ImPlot::EndPlot();
      }

      ImPlot::PopColormap();
    }
  }
}


void ElementProfiler::renderPieChartNode(const EntryNode& node, int level, int numLevels, double plotRadius, double angle0, double totalTime)
{
  // Gather data
  std::vector<const char*> labels(node.child.size());
  std::vector<float>       data(node.child.size());
  for(size_t i = 0; i < node.child.size(); i++)
  {
    labels[i] = node.child[i].name.c_str();
    data[i]   = static_cast<float>(node.child[i].gpuTime / totalTime);
  }

  // The 0.5 makes it so that the bottom level is at half radius, so that we
  // can see the wedges clearly.
  const double myRadius = numLevels == 1 ? plotRadius : plotRadius * (1.0 - (0.5 * level) / (numLevels - 1));

  // Since ImPlot always draws text at half radius -- which gets covered up if
  // there are multiple levels -- only draw timers if this is the bottom level.
  const char* text = (level + 1 == numLevels ? "%.2f" : "");
  ImPlot::PlotPieChart(labels.data(), data.data(), static_cast<int>(data.size()), 0.5, 0.5, myRadius, text, angle0);

  // Recurse over children
  if(level + 1 < numLevels && node.child.size() > 0)
  {
    for(size_t i = 0; i < node.child.size(); i++)
    {
      renderPieChartNode(node.child[i], level + 1, numLevels, plotRadius, angle0, totalTime);
      angle0 += 360.0 * node.child[i].gpuTime / totalTime;
    }
  }
}


void ElementProfiler::renderBarChart(View& view)
{
  const bool  gridMode = ImGui::GetContentRegionAvail().x >= 600 && m_frameNodes.size() > 1;
  const float width    = ImGui::GetContentRegionAvail().x / (gridMode ? 2.0f : 1.0f) - 5.0f;

  drawVsyncCheckbox();

  ImGui::SameLine();
  ImGui::CheckboxFlags("Stacked", (unsigned int*)&view.state->barChart.stacked, ImPlotBarGroupsFlags_Stacked);

  for(auto i = 0; i < m_frameNodes.size(); ++i)
  {
    const auto& rootNode = m_frameNodes[i];

    // each root node is a timeline
    if(!rootNode.child.empty())
    {
      const auto& node   = rootNode.child[0];
      int         items  = 0;
      const int   groups = 1;
      const float size   = 0.67f;

      // Get all Level 0 values
      std::vector<const char*> labels1(node.child.size());
      std::vector<float>       data1(node.child.size());
      for(size_t i = 0; i < node.child.size(); i++)
      {
        items++;
        labels1[i] = node.child[i].name.c_str();
        data1[i]   = node.child[i].gpuTime;
      }

      if(gridMode && i % 2 != 0)
        ImGui::SameLine();

      // use different color palette for better legibility
      if(i % 3 == 0)
        ImPlot::PushColormap(ImPlotColormap_Deep);
      if(i % 3 == 1)
        ImPlot::PushColormap(ImPlotColormap_Pastel);
      if(i % 3 == 2)
        ImPlot::PushColormap(ImPlotColormap_Viridis);

      if(ImPlot::BeginPlot(rootNode.name.c_str(), ImVec2(width, (float)view.state->plotHeight), ImPlotFlags_NoMouseText))
      {
        ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
        ImPlot::SetupAxes("Time in milliseconds", "Timers", ImPlotAxisFlags_AutoFit,
                          ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoTickMarks
                              | ImPlotAxisFlags_NoGridLines);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -0.4, 0.6, ImPlotCond_Always);
        // If `items` is empty. ImPlot divides by 0. In this case, we choose to
        // show an empty plot instead.
        if(items > 0)
        {
          ImPlot::PlotBarGroups(labels1.data(), data1.data(), items, groups, size, 0,
                                view.state->barChart.stacked | ImPlotBarGroupsFlags_Horizontal);
        }
        ImPlot::EndPlot();
      }

      ImPlot::PopColormap();
    }
  }
}

//
void ElementProfiler::renderLineChart(View& view)
{
  const bool  gridMode = ImGui::GetContentRegionAvail().x >= 600 && m_frameNodes.size() > 1;
  const float width    = ImGui::GetContentRegionAvail().x / (gridMode ? 2.0f : 1.0f) - 5.0f;

  drawVsyncCheckbox();
  ImGui::SameLine();
  ImGui::Checkbox("CPU line", &view.state->lineChart.cpuLine);
  ImGui::SameLine();
  ImGui::Checkbox("GPU lines", &view.state->lineChart.gpuLines);
  ImGui::SameLine();
  ImGui::Checkbox("GPU fills", &view.state->lineChart.gpuFills);

  for(auto i = 0; i < m_frameNodes.size(); ++i)
  {
    const auto& rootNode = m_frameNodes[i];

    if(!rootNode.child.empty())
    {
      const auto& node = rootNode.child[0];

      std::vector<const char*>        gpuTimesLabels(node.child.size());
      std::vector<std::vector<float>> gpuTimes(node.child.size());
      std::vector<float>              cpuTimes(node.timerInfo.numAveraged);
      float                           avgCpuTime = 0.f;
      float                           avgGpuTime = 0.f;

      // filling the GPU times for each timer
      for(size_t i = 0; i < node.child.size(); i++)
      {
        const auto& child      = node.child[i];
        float       gpuTimeSum = 0.f;

        gpuTimesLabels[i] = child.name.c_str();

        if(!child.timerInfo.gpu.times.empty())
        {
          gpuTimes[i].resize(child.timerInfo.numAveraged);
          for(size_t j = 0; j < child.timerInfo.numAveraged; j++)
          {
            uint32_t index = (child.timerInfo.gpu.index - child.timerInfo.numAveraged + j) % nvutils::ProfilerTimeline::MAX_LAST_FRAMES;
            gpuTimes[i][j] = float(child.timerInfo.gpu.times[index] / 1000.0);

            if(i > 0)
            {
              gpuTimes[i][j] += gpuTimes[i - 1][j];
            }

            gpuTimeSum += gpuTimes[i][j];
          }

          if(child.timerInfo.numAveraged != 0)
            avgGpuTime += gpuTimeSum / child.timerInfo.numAveraged;
        }
      }

      // filling the top level CPU times
      for(size_t j = 0; j < node.timerInfo.numAveraged; j++)
      {
        uint32_t index = (node.timerInfo.cpu.index - node.timerInfo.numAveraged + j) % nvutils::ProfilerTimeline::MAX_LAST_FRAMES;
        cpuTimes[j] = float(node.timerInfo.cpu.times[index] / 1000.0);
        avgCpuTime += cpuTimes[j];
      }

      float avgTime = 0.f;

      if(node.timerInfo.numAveraged > 0)
      {
        avgCpuTime /= node.timerInfo.numAveraged;
        avgTime = view.state->lineChart.cpuLine ? avgCpuTime : avgGpuTime;
      }
      if(view.maxY == 0.f)
      {
        view.maxY = avgTime;
      }
      else
      {
        const float PROFILER_GRAPH_TEMPORAL_SMOOTHING = 20.f;
        view.maxY = (PROFILER_GRAPH_TEMPORAL_SMOOTHING * view.maxY + avgTime) / (PROFILER_GRAPH_TEMPORAL_SMOOTHING + 1.f);
      }

      // If there is something top plot: Let's plot !
      if(gpuTimes.size() > 0 && gpuTimes[0].size() > 0)
      {

        if(gridMode && i % 2 != 0)
          ImGui::SameLine();

        // use different color palette for better legibility
        if(i % 3 == 0)
          ImPlot::PushColormap(ImPlotColormap_Deep);
        if(i % 3 == 1)
          ImPlot::PushColormap(ImPlotColormap_Pastel);
        if(i % 3 == 2)
          ImPlot::PushColormap(ImPlotColormap_Viridis);

        const ImPlotFlags     plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs;
        const ImPlotAxisFlags axesFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel;

        if(ImPlot::BeginPlot(rootNode.name.c_str(), ImVec2(width, (float)view.state->plotHeight), plotFlags))
        {
          ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
          ImPlot::SetupAxes(nullptr, "Count", axesFlags | ImPlotAxisFlags_NoTickLabels, axesFlags);
          ImPlot::SetupAxesLimits(0, node.child[0].timerInfo.numAveraged, 0, view.maxY * 1.2, ImPlotCond_Always);

          ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);

          if(view.state->lineChart.cpuLine)
          {
            ImPlot::SetNextLineStyle(ImColor(1.f, 0.f, 0.f, 1.0f), 0.1f);
            ImPlot::PlotLine("CPU", cpuTimes.data(), (int)cpuTimes.size());
          }

          ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);

          for(size_t i = 0; i < node.child.size(); i++)
          {
            size_t index = node.child.size() - i - 1;
            if(view.state->lineChart.gpuFills)
            {
              ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
              ImPlot::PlotShaded(node.child[index].name.c_str(), gpuTimes[index].data(), (int)gpuTimes[index].size(),
                                 -INFINITY, 1.0, 0, 0, 0);
              ImPlot::PopStyleVar();
            }
            if(view.state->lineChart.gpuLines)
            {
              ImPlot::PlotLine(node.child[index].name.c_str(), gpuTimes[index].data(), (int)gpuTimes[index].size());
            }
          }

          if(ImPlot::IsPlotHovered())
          {
            ImPlotPoint        mouse       = ImPlot::GetPlotMousePos();
            int                mouseOffset = (int(mouse.x)) % (int)gpuTimes[0].size();
            std::vector<float> localTimes(node.child.size());
            ImGui::BeginTooltip();

            ImGui::Text("CPU: %.3f ms", cpuTimes[mouseOffset]);

            float totalGpu = 0.f;
            for(size_t i = 0; i < node.child.size(); i++)
            {
              if(i == 0)
              {
                localTimes[i] = gpuTimes[i][mouseOffset];
              }
              else
              {
                localTimes[i] = gpuTimes[i][mouseOffset] - gpuTimes[i - 1][mouseOffset];
              }
              totalGpu += localTimes[i];
            }
            ImGui::Text("GPU: %.3f ms", totalGpu);
            for(size_t i = 0; i < node.child.size(); i++)
            {
              ImGui::Text("  %s: %.3f ms (%.1f%%)", node.child[i].name.c_str(), localTimes[i], localTimes[i] * 100.f / totalGpu);
            }

            ImGui::EndTooltip();
          }

          ImPlot::EndPlot();
        }

        ImPlot::PopColormap();
      }
    }
  }
}

void ElementProfiler::addSettingsHandler()
{

  // finds the wubsection in wich we are
  auto readOpen = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void* {
    auto* self = static_cast<ElementProfiler*>(handler->UserData);
    // Identify which view subsection is being read and return its index
    for(size_t i = 0; i < self->m_views.size(); ++i)
    {
      if(strcmp(name, self->m_views[i].state->name.c_str()) == 0)
        return (void*)(i + 1);
    }
    return nullptr;
  };

  // Save settings handler, not using capture so can be used as a function pointer
  auto saveAllToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
    auto* self = static_cast<ElementProfiler*>(handler->UserData);
    for(const auto& view : self->m_views)
    {
      buf->appendf("[%s][%s]\n", handler->TypeName, view.state->name.c_str());
      buf->appendf("ShowWindow=%d\n", view.state->show ? 1 : 0);
    }
    buf->append("\n");
  };

  // Load settings handler, not using capture so can be used as a function pointer
  auto loadLineFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
    intptr_t view_id = (intptr_t)entry - 1;
    auto*    self    = static_cast<ElementProfiler*>(handler->UserData);
    int      value;

#ifdef _MSC_VER
    if(sscanf_s(line, "ShowWindow=%d", &value) == 1)
#else
    if(sscanf(line, "ShowWindow=%d", &value) == 1)
#endif
    {
      self->m_views[view_id].state->show = (value == 1);
    }
  };

  //
  ImGuiSettingsHandler iniHandler;
  iniHandler.TypeName   = "ElementProfiler";
  iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
  iniHandler.ReadOpenFn = readOpen;
  iniHandler.WriteAllFn = saveAllToIni;
  iniHandler.ReadLineFn = loadLineFromIni;
  iniHandler.UserData   = this;  // Pass the current instance to the handler
  ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
}
