#include <algorithm>
#include <filter/SkeletonFilter.hpp>
#include <filter/Utils.hpp>
#include <filter/adaptive/AdaptiveBarShalomFilter1D.hpp>
#include <filter/adaptive/AdaptiveConstrainedSkeletonFilter.hpp>
#include <filter/adaptive/AdaptivePointFilter3D.hpp>
#include <filter/adaptive/AdaptiveZarchanFilter1D.hpp>
#include <gtest/gtest.h>

typedef AdaptivePointFilter3D<double, AdaptiveZarchanFilter1D<double>> ZarPointFilter;

TEST(TestSaverSkeleton, BasicAssertions)
{
    auto cached_var = get_cached_measurement_error();
    std::string data_path(std::format("/home/{}/repos/master/code/_matlab/sts_NFOV_UNBINNED_720P_30fps.json", std::getenv("USER")));

    auto joint_count = 32;
    auto [joints, n_frames, timestamps, is_null] = load_data(data_path, joint_count, 870);

    std::vector<Point<double>> initial_points;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_points.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }

    AdaptiveConstrainedSkeletonFilter<double, ZarPointFilter> filter(32, cached_var, get_azure_kinect_com_matrix());
    filter.init(initial_points, timestamps[0]);

    std::vector<std::vector<Point<double>>> filtered_values;
    // Add initial value
    std::vector<Point<double>> initial_joints;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_joints.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }

    // Only go ten iteration
    int max_frame = 10;
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

    auto _json = filter.to_json();
    EXPECT_EQ(_json["timestamps"].size(), max_frame - 1);
    EXPECT_EQ(_json["unfiltered_positions"].size(), max_frame - 1);
    EXPECT_EQ(_json["filtered_positions"].size(), max_frame - 1);
    EXPECT_EQ(_json["filtered_velocities"].size(), max_frame - 1);

    json auto_json = filter;
    EXPECT_EQ(auto_json["timestamps"].size(), max_frame - 1);
    EXPECT_EQ(auto_json["unfiltered_positions"].size(), max_frame - 1);
    EXPECT_EQ(auto_json["filtered_positions"].size(), max_frame - 1);
    EXPECT_EQ(auto_json["filtered_velocities"].size(), max_frame - 1);

    EXPECT_TRUE(filter.get_filtered_positions() == filtered_values);
}

TEST(TestSaverToJsonWithMultipleSkeletons, BasicAssertions)
{
    SkeletonFilterBuilder<double> builder(32, 2.0);
    std::vector<SkeletonFilter<double>> filters;
    int amount = 5;
    for (int i = 0; i < amount; ++i) {
        filters.push_back(builder.build());
    }

    std::string data_path(std::format("/home/{}/repos/master/code/_matlab/sts_NFOV_UNBINNED_720P_30fps.json", std::getenv("USER")));

    auto joint_count = 32;
    auto [joints, n_frames, timestamps, is_null] = load_data(data_path, joint_count, 870);
    std::vector<Point<double>> initial_points;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_points.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }

    // Only go ten iterations
    int max_frame = 10;

    // Do this for each filter
    // Take a ref to the filter s.t. we do not copy it
    for (auto& filter : filters) {

        filter.init(initial_points, timestamps[0]);

        std::vector<Point<double>> initial_joints;
        for (int joint = 0; joint < joint_count; ++joint) {
            initial_joints.push_back(Point<double>(
                joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
        }
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

            auto _ = filter.step(current_joint_positions, timestamps[frame_idx]);
        }
    }

    json _json;
    _json["filters"] = filters;

    for (int i = 0; i < amount; ++i) {
        EXPECT_EQ(_json["filters"][i]["timestamps"].size(), max_frame - 1);
        EXPECT_EQ(_json["filters"][i]["unfiltered_positions"].size(), max_frame - 1);
        EXPECT_EQ(_json["filters"][i]["filtered_positions"].size(), max_frame - 1);
        EXPECT_EQ(_json["filters"][i]["filtered_velocities"].size(), max_frame - 1);
    }
}
