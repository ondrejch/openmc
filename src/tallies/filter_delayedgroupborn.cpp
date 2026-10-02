#include "openmc/tallies/filter_delayedgroupborn.h"

#include <stdexcept> // for invalid_argument

#include <fmt/core.h>

#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// DelayedGroupBornFilter implementation
//==============================================================================

void DelayedGroupBornFilter::from_xml(pugi::xml_node node)
{
  auto groups = get_node_array<int>(node, "bins");
  this->set_groups(groups);
}

void DelayedGroupBornFilter::set_groups(span<const int> groups)
{
  // Validate all groups before changing the filter, so that an invalid list
  // leaves the filter unchanged
  std::unordered_map<int, int> map;
  for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
    int group = groups[i];
    if (group < 0 || group > MAX_DELAYED_GROUPS) {
      throw std::invalid_argument {fmt::format(
        "Encountered delayedgroupborn bin with group {}, which is outside the "
        "range 0 (prompt-born) to MAX_DELAYED_GROUPS ({}).",
        group, MAX_DELAYED_GROUPS)};
    }
    if (!map.emplace(group, i).second) {
      throw std::invalid_argument {fmt::format(
        "Encountered delayedgroupborn bin with group {} more than once.",
        group)};
    }
  }

  groups_.assign(groups.begin(), groups.end());
  map_ = std::move(map);
  n_bins_ = groups_.size();
}

void DelayedGroupBornFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  // The birth delayed group is set when the particle is started from its
  // source site and does not change along the history. p.delayed_group() is
  // not used because multigroup fission site creation overwrites it.
  auto search = map_.find(p.delayed_group_born());
  if (search != map_.end()) {
    match.bins_.push_back(search->second);
    match.weights_.push_back(1.0);
  }
}

void DelayedGroupBornFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "bins", groups_);
}

std::string DelayedGroupBornFilter::text_label(int bin) const
{
  int group = groups_[bin];
  if (group == 0) {
    return "Born Prompt";
  }
  return fmt::format("Born from Delayed Group {}", group);
}

} // namespace openmc
