//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// reference& operator=(bool)

#include <cassert>
#include <vector>

bool test() {
  std::vector<bool> vec;
  typedef std::vector<bool>::reference Ref;
  vec.push_back(true);
  vec.push_back(false);
  Ref ref = vec[0];
  const Ref cref = vec[0];

  assert(ref);
  ref = false;
  assert(!vec[0]);
  assert(!vec[1]);
  ref = true;
  assert(vec[0]);
  assert(!vec[1]);

  assert(cref);

  return true;
}

int main(int, char**) {
  test();

  return 0;
}
