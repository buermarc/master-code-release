#include <Eigen/Dense>
#include <filter/ConstrainedSkeletonFilter.hpp>
#include <filter/PointFilter3D.hpp>
#include <filter/Utils.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

#include <fstream>
#include <iostream>

#include <filter/ConstrainedSkeletonFilter.hpp>
#include <filter/SkeletonFilter.hpp>
#include <filter/Utils.hpp>
#include <filter/adaptive/AdaptiveBarShalomFilter1D.hpp>
#include <filter/adaptive/AdaptiveConstrainedSkeletonFilter.hpp>
#include <filter/adaptive/AdaptivePointFilter3D.hpp>
#include <filter/adaptive/AdaptiveRoseFilter1D.hpp>
#include <filter/adaptive/AdaptiveZarchanFilter1D.hpp>
#include <filter/com.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

// include this header to serialize vectors
#include <boost/serialization/vector.hpp>

typedef AdaptivePointFilter3D<double, AdaptiveRoseFilter1D<double>> RosePointFilter;
typedef AdaptivePointFilter3D<double, AdaptiveBarShalomFilter1D<double>> BarPointFilter;
typedef AdaptivePointFilter3D<double, AdaptiveZarchanFilter1D<double>> ZarPointFilter;

template <typename T>
std::vector<T> flatten(std::vector<std::vector<T>> const& vec)
{
    std::vector<T> flattened;
    for (auto const& v : vec) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
}

TEST(TestCachedMeasurementError, BasicAssertions)
{
    std::stringstream ss;
    ss << "/home/";
    ss << std::getenv("USER");
    ss << "/repos/master/code/_matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json";
    std::string var_path(ss.str());
    int joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);
    auto cached_var = get_cached_measurement_error();
    // Only this much significant values are cached
    EXPECT_TRUE(var.isApprox(cached_var, 0.00001));
}

template <typename FilterType>
void test_adaptive_constrained_skeleton_filter(std::string name)
{
    auto cached_var = get_cached_measurement_error();
    std::stringstream ssd;
    ssd << "/home/";
    ssd << std::getenv("USER");
    ssd << "/repos/master/code/matlab/sts_NFOV_UNBINNED_720P_30fps.json";
    std::string data_path(ssd.str());

    auto joint_count = 32;
    auto [joints, n_frames, timestamps, is_null] = load_data(data_path, joint_count, 870);

    std::vector<Point<double>> initial_points;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_points.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }

    AdaptiveConstrainedSkeletonFilter<double, FilterType> filter(32, cached_var);
    filter.init(initial_points, timestamps[0]);

    std::vector<std::vector<Point<double>>> filtered_values;
    // Add initial value
    std::vector<Point<double>> initial_joints;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_joints.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }
    filtered_values.push_back(initial_joints);

    // Only go one iteration
    int max_frame = 2;
    //
    // Make sure we have enough data
    EXPECT_TRUE(max_frame <= n_frames);

    for (int frame_idx = 1; frame_idx < max_frame; ++frame_idx) {
        if (is_null[frame_idx])
            continue;

        std::vector<Point<double>> current_joint_positions;
        for (int joint = 0; joint < joint_count; ++joint) {
            current_joint_positions.push_back(Point<double>(
                joints(frame_idx, joint, 0), joints(frame_idx, joint, 1),
                joints(frame_idx, joint, 2)));
        }

        auto [values, _] = filter.step(current_joint_positions, timestamps[frame_idx]);
        filtered_values.push_back(values);
    }

    std::vector<std::vector<Point<double>>> expected_values;
    {
        std::stringstream ss;
        ss << std::getenv("ASSETS_DIR") << name << ".dat";
        std::ifstream ifs(ss.str());
        boost::archive::text_iarchive ia(ifs);
        ia& expected_values;
    }

    EXPECT_TRUE(filtered_values == expected_values);
}

TEST(AdaptiveConstrainedSkeletonFilterBasicSanityCheckZarPointFilter, BasicAssertions)
{
    test_adaptive_constrained_skeleton_filter<ZarPointFilter>("zar");
}

TEST(AdaptiveConstrainedSkeletonFilterBasicSanityCheckBarPointFilter, BasicAssertions)
{
    test_adaptive_constrained_skeleton_filter<BarPointFilter>("bar");
}
TEST(AdaptiveConstrainedSkeletonFilterBasicSanityCheckRosePointFilter, BasicAssertions)
{
    test_adaptive_constrained_skeleton_filter<RosePointFilter>("rose_point");
}
