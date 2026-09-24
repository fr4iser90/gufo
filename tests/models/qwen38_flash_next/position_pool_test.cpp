#include "src/models/qwen38_flash_next/kernels/rocm/position_pool.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using gufo::models::qwen38_flash_next::rocm::PositionPool;

void Expect(bool ok, const char* message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

void TestReserveReleaseCoalesce() {
  PositionPool pool(8);
  Expect(pool.capacity() == 8, "capacity");
  Expect(pool.free_positions() == 8, "initially free");

  std::uint32_t a = 0;
  std::uint32_t b = 0;
  Expect(pool.Reserve(3, &a), "reserve a");
  Expect(a == 0, "a at base");
  Expect(pool.Reserve(2, &b), "reserve b");
  Expect(b == 3, "b follows a");
  Expect(pool.used() == 5, "used after two reserves");
  Expect(!pool.Reserve(4, &a), "no room for 4");

  pool.Release(0, 3);
  Expect(pool.used() == 2, "used after release a");
  // Free [0,3) and keep [5,3); cannot take 4 contiguous yet.
  Expect(!pool.Reserve(4, &a), "gap blocks four");
  pool.Release(b, 2);
  Expect(pool.free_positions() == 8, "fully free after coalesce");
  Expect(pool.Reserve(8, &a), "full pool after coalesce");
  Expect(a == 0, "full reservation at base");
}

void TestTryGrow() {
  PositionPool pool(16);
  std::uint32_t base = 0;
  Expect(pool.Reserve(4, &base), "seed");
  Expect(pool.TryGrow(base, 4, 10), "grow in place");
  Expect(pool.used() == 10, "grown length");
  Expect(!pool.TryGrow(base, 10, 17), "past capacity");

  std::uint32_t other = 0;
  Expect(pool.Reserve(4, &other), "tail reservation");
  Expect(other == 10, "tail starts after grown span");
  Expect(!pool.TryGrow(base, 10, 12), "blocked by neighbour");
}

void TestHalogenStyleTwoSlotsSharePool() {
  // One full native context shared by two slots: only one full reservation
  // fits; a shorter second conversation still fits beside it.
  constexpr std::uint32_t kCtx = 262144;
  PositionPool pool(kCtx);
  std::uint32_t long_chat = 0;
  std::uint32_t short_chat = 0;
  Expect(pool.Reserve(kCtx, &long_chat), "full conversation");
  Expect(!pool.Reserve(kCtx, &short_chat), "second full denied");
  pool.Release(long_chat, kCtx);
  Expect(pool.Reserve(200000, &long_chat), "long region");
  Expect(pool.Reserve(60000, &short_chat), "short neighbour");
  Expect(pool.free_positions() == 2144, "remaining headroom");
}

}  // namespace

int main() {
  try {
    TestReserveReleaseCoalesce();
    TestTryGrow();
    TestHalogenStyleTwoSlotsSharePool();
  } catch (const std::exception& error) {
    return 1;
  }
  return 0;
}
