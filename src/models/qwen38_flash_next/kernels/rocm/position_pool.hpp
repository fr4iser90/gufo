#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_POSITION_POOL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_POSITION_POOL_HPP_

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace gufo::models::qwen38_flash_next::rocm {

/// Contiguous freelist over a fixed token-position capacity. Halogen-style
/// shared KV uses one pool for every conversation; slots reserve spans at
/// admission / growth rather than owning a private full-context arena.
class PositionPool {
public:
  explicit PositionPool(std::uint32_t capacity) : capacity_(capacity) {
    if (capacity_ > 0) {
      free_.push_back({0, capacity_});
    }
  }

  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::uint32_t used() const noexcept { return used_; }
  [[nodiscard]] std::uint32_t free_positions() const noexcept {
    return capacity_ - used_;
  }

  /// First-fit contiguous reservation. Returns false when no free span holds
  /// `length` positions.
  [[nodiscard]] bool Reserve(std::uint32_t length, std::uint32_t* base) {
    if (base == nullptr || length == 0 || length > capacity_ - used_) {
      return false;
    }
    for (std::size_t i = 0; i < free_.size(); ++i) {
      auto& region = free_[i];
      if (region.second < length) {
        continue;
      }
      *base = region.first;
      region.first += length;
      region.second -= length;
      if (region.second == 0) {
        free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(i));
      }
      used_ += length;
      return true;
    }
    return false;
  }

  void Release(std::uint32_t base, std::uint32_t length) {
    if (length == 0) {
      return;
    }
    free_.push_back({base, length});
    std::sort(free_.begin(), free_.end(), [](const Region& a, const Region& b) {
      return a.first < b.first;
    });
    std::vector<Region> coalesced;
    coalesced.reserve(free_.size());
    for (const auto& region : free_) {
      if (!coalesced.empty() &&
          coalesced.back().first + coalesced.back().second == region.first) {
        coalesced.back().second += region.second;
      } else {
        coalesced.push_back(region);
      }
    }
    free_ = std::move(coalesced);
    used_ -= length;
  }

  /// Extends an existing span in place when the following positions are free.
  [[nodiscard]] bool TryGrow(std::uint32_t base, std::uint32_t old_length,
                             std::uint32_t new_length) {
    if (new_length < old_length ||
        new_length - old_length > capacity_ - used_) {
      return false;
    }
    if (new_length == old_length) {
      return true;
    }
    const std::uint32_t need = new_length - old_length;
    const std::uint32_t edge = base + old_length;
    for (std::size_t i = 0; i < free_.size(); ++i) {
      auto& region = free_[i];
      if (region.first != edge || region.second < need) {
        continue;
      }
      region.first += need;
      region.second -= need;
      if (region.second == 0) {
        free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(i));
      }
      used_ += need;
      return true;
    }
    return false;
  }

private:
  using Region = std::pair<std::uint32_t, std::uint32_t>;

  std::uint32_t capacity_{0};
  std::uint32_t used_{0};
  std::vector<Region> free_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_POSITION_POOL_HPP_
