// Minimal assertions for the tests; each test binary exits non-zero on failure.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace mlxpipe_test {
inline int& failures() {
    static int count = 0;
    return count;
}
inline std::string source_dir() { return MLXPIPE_SOURCE_DIR; }
}  // namespace mlxpipe_test

#define EXPECT_TRUE(cond)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++mlxpipe_test::failures();                                            \
        }                                                                          \
    } while (0)

#define EXPECT_EQ(a, b)                                                                    \
    do {                                                                                   \
        auto va_ = (a);                                                                    \
        auto vb_ = (b);                                                                    \
        if (!(va_ == vb_)) {                                                               \
            std::fprintf(stderr, "%s:%d: expected %s == %s\n", __FILE__, __LINE__, #a, #b); \
            ++mlxpipe_test::failures();                                                    \
        }                                                                                  \
    } while (0)

#define EXPECT_NEAR(a, b, tol)                                                                          \
    do {                                                                                                \
        double va_ = (a), vb_ = (b);                                                                    \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                                         \
            std::fprintf(stderr, "%s:%d: expected %s ~= %s (%g vs %g, tol %g)\n", __FILE__, __LINE__, #a, #b, \
                         va_, vb_, (double)(tol));                                                      \
            ++mlxpipe_test::failures();                                                                 \
        }                                                                                               \
    } while (0)

#define TEST_MAIN_END()                                                         \
    do {                                                                        \
        if (mlxpipe_test::failures()) {                                         \
            std::fprintf(stderr, "%d failure(s)\n", mlxpipe_test::failures());  \
            return 1;                                                           \
        }                                                                       \
        std::printf("OK\n");                                                    \
        return 0;                                                               \
    } while (0)
