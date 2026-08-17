#include "gemma_encoder.h"
#include "tensor.h"

#include <cassert>

int main() {
    // 10 rows x 2 cols, row r = [2r, 2r+1], so each row is identifiable by value.
    Tensor input({10, 2});
    for (size_t r = 0; r < 10; ++r) {
        input.at({r, 0}) = static_cast<float>(2 * r);
        input.at({r, 1}) = static_cast<float>(2 * r + 1);
    }

    // select_index = [0] + [-3,-2,-1] = [0, 7, 8, 9]
    Tensor kept = select_rows_first_and_last(input, 4);
    assert(kept.dim_size(0) == 4);
    assert(kept.dim_size(1) == 2);
    assert(kept.at({0, 0}) == 0.0f && kept.at({0, 1}) == 1.0f);   // row 0
    assert(kept.at({1, 0}) == 14.0f && kept.at({1, 1}) == 15.0f); // row 7
    assert(kept.at({2, 0}) == 16.0f && kept.at({2, 1}) == 17.0f); // row 8
    assert(kept.at({3, 0}) == 18.0f && kept.at({3, 1}) == 19.0f); // row 9

    // Edge case: keep == rows (identity).
    Tensor identity = select_rows_first_and_last(input, 10);
    for (size_t r = 0; r < 10; ++r) {
        assert(identity.at({r, 0}) == input.at({r, 0}));
        assert(identity.at({r, 1}) == input.at({r, 1}));
    }

    // Edge case: keep == 1 (row 0 only, empty tail range).
    Tensor single = select_rows_first_and_last(input, 1);
    assert(single.dim_size(0) == 1);
    assert(single.at({0, 0}) == 0.0f && single.at({0, 1}) == 1.0f);

    return 0;
}
