#ifndef OPENMC_TALLIES_FILTER_DELAYEDGROUPBORN_H
#define OPENMC_TALLIES_FILTER_DELAYEDGROUPBORN_H

#include <string>
#include <unordered_map>

#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Bins events by the delayed group that the particle was born from.
//!
//! The bin of an event is given by Particle::delayed_group_born(), the delayed
//! group of the source site that the particle was started from: 0 for a prompt
//! fission neutron and for any site that is not a fission site, and g = 1, ...,
//! MAX_DELAYED_GROUPS for a delayed neutron emitted by a precursor of group g.
//! Particle::delayed_group() is not used because multigroup fission site
//! creation overwrites it. The value is constant along a history, so the
//! filter works with any estimator. Secondary particles, such as the extra
//! neutrons of (n,xn) reactions, and particles created by splitting are
//! started from sites that carry group 0, so they count as prompt-born; the
//! birth position used by MeshBornFilter has the same limitation. Unlike
//! DelayedGroupFilter, which bins the fission neutrons produced in an event,
//! this filter bins the event itself and needs no special scoring.
//==============================================================================

class DelayedGroupBornFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~DelayedGroupBornFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "delayedgroupborn"; }
  FilterType type() const override { return FilterType::DELAYED_GROUP_BORN; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  const vector<int>& groups() const { return groups_; }

  //! Set the birth delayed groups, one per bin
  //
  //! \param[in] groups  Distinct groups in [0, MAX_DELAYED_GROUPS], where 0
  //!    means prompt-born. An invalid list leaves the filter unchanged.
  void set_groups(span<const int> groups);

private:
  //----------------------------------------------------------------------------
  // Data members

  //! Birth delayed group of each bin (0 = prompt-born)
  vector<int> groups_;

  //! A map from birth delayed group to filter bin index
  std::unordered_map<int, int> map_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_DELAYEDGROUPBORN_H
