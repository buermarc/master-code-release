#include <Eigen/Dense>
#include <filter/ConstrainedSkeletonFilter.hpp>
#include <filter/PointFilter3D.hpp>
#include <filter/Utils.hpp>
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

TEST(TestCachedMeasurementError, BasicAssertions)
{
    std::string var_path("../matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json");
    int joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);
    auto cached_var = get_cached_measurement_error();
    // Only this much significant values are cached
    EXPECT_TRUE(var.isApprox(cached_var, 0.00001));
}
