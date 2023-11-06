#include <Eigen/Dense>
#include <filter/ConstrainedSkeletonFilter.hpp>
#include <filter/PointFilter3D.hpp>
#include <filter/Utils.hpp>
#include <gtest/gtest.h>

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


typedef AdaptivePointFilter3D<double, AdaptiveRoseFilter1D<double>> RosePointFilter;
typedef AdaptivePointFilter3D<double, AdaptiveBarShalomFilter1D<double>> BarPointFilter;
typedef AdaptivePointFilter3D<double, AdaptiveZarchanFilter1D<double>> ZarPointFilter;

template<typename T>
std::vector<T> flatten(std::vector<std::vector<T>> const &vec)
{
    std::vector<T> flattened;
    for (auto const &v: vec) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
}

TEST(TestCachedMeasurementError, BasicAssertions)
{
    std::string var_path(std::format("/home/{}/repos/master/code/matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json", std::getenv("USER")));
    int joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);
    auto cached_var = get_cached_measurement_error();
    // Only this much significant values are cached
    EXPECT_TRUE(var.isApprox(cached_var, 0.00001));
}

template<typename FilterType>
void test_adaptive_constrained_skeleton_filter(std::vector<Point<double>> expected)
{
    auto cached_var = get_cached_measurement_error();
    std::string data_path(std::format("/home/{}/repos/master/code/matlab/sts_NFOV_UNBINNED_720P_30fps.json", std::getenv("USER")));

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

    std::cout << "{";
    for (auto value : filtered_values) {
        for (auto joint : value) {
            std::cout << "{";
            std::cout << joint.x << ",";
            std::cout << joint.y << ",";
            std::cout << joint.z;
            std::cout << "},";
        }
    }
    std::cout << "}";

    auto flat = flatten(filtered_values);
    EXPECT_TRUE(flat.size() == expected.size());
    for (int i = 0; i < flat.size(); ++i) {
        EXPECT_TRUE(flat[i] == expected[i]);
    }
}

/*
TEST(AdaptiveConstrainedSkeletonFilterBasicSanityCheckZarPointFilter, BasicAssertions)
{
    std::vector<Point<double>> expected = {{-0.23315,0.171265,2.17361},{-0.229523,0.00518617,2.19278},{-0.225886,-0.128381,2.1885},{-0.223162,-0.331447,2.17561},{-0.192146,-0.29794,2.1807},{-0.0636616,-0.260943,2.18943},{-0.0327383,-0.00795043,2.23557},{-0.0387984,0.197247,2.16403},{0.0173699,0.267136,2.1537},{-0.0253986,0.321155,2.07947},{-0.0703645,0.288735,2.08823},{-0.255099,-0.297973,2.17674},{-0.373681,-0.266332,2.16527},{-0.429734,-0.0101757,2.19518},{-0.426251,0.198687,2.12569},{-0.462696,0.284368,2.15352},{-0.463397,0.352181,2.08064},{-0.408895,0.280529,2.04919},{-0.147771,0.17349,2.17673},{-0.100429,0.391397,1.87494},{-0.125154,0.735275,1.97362},{-0.076999,0.833148,1.83602},{-0.31014,0.169258,2.17079},{-0.319744,0.384523,1.86409},{-0.313362,0.734925,1.95627},{-0.344045,0.828273,1.82804},{-0.221365,-0.40601,2.1583},{-0.159357,-0.425139,2.02239},{-0.145486,-0.458999,2.05998},{-0.138936,-0.455417,2.17857},{-0.188823,-0.460705,2.03717},{-0.280763,-0.473882,2.11751},{-0.233158,0.176178,2.1798},{-0.229384,0.00722323,2.19077},{-0.226015,-0.127698,2.18004},{-0.222712,-0.333658,2.1736},{-0.191358,-0.299481,2.1773},{-0.0616837,-0.263714,2.19516},{-0.0320325,-0.00483526,2.22539},{-0.0295941,0.200511,2.14601},{0.019159,0.275206,2.16601},{-0.0216646,0.330557,2.08991},{-0.024852,0.295904,2.05497},{-0.255155,-0.299835,2.17402},{-0.376716,-0.2715,2.17201},{-0.425043,-0.0092676,2.1904},{-0.428321,0.200767,2.11568},{-0.47454,0.280073,2.15117},{-0.460702,0.351005,2.08087},{-0.411754,0.293246,2.04165},{-0.146641,0.178213,2.18138},{-0.0984112,0.390059,1.8696},{-0.116921,0.737016,1.97533},{-0.0752415,0.83133,1.83044},{-0.311173,0.174343,2.17838},{-0.319111,0.379891,1.85933},{-0.312389,0.733574,1.95708},{-0.345933,0.827549,1.82741},{-0.220835,-0.409693,2.1585},{-0.148371,-0.430172,2.02586},{-0.135592,-0.462874,2.06574},{-0.137073,-0.455837,2.18586},{-0.177757,-0.466937,2.03988},{-0.275485,-0.481812,2.11535}};
    test_adaptive_constrained_skeleton_filter<ZarPointFilter>(expected);
}


TEST(AdaptiveConstrainedSkeletonFilterBasicSanityCheckBarPointFilter, BasicAssertions)
{
    std::vector<Point<double>> expected = {{-0.23315,0.171265,2.17361},{-0.229523,0.00518617,2.19278},{-0.225886,-0.128381,2.1885},{-0.223162,-0.331447,2.17561},{-0.192146,-0.29794,2.1807},{-0.0636616,-0.260943,2.18943},{-0.0327383,-0.00795043,2.23557},{-0.0387984,0.197247,2.16403},{0.0173699,0.267136,2.1537},{-0.0253986,0.321155,2.07947},{-0.0703645,0.288735,2.08823},{-0.255099,-0.297973,2.17674},{-0.373681,-0.266332,2.16527},{-0.429734,-0.0101757,2.19518},{-0.426251,0.198687,2.12569},{-0.462696,0.284368,2.15352},{-0.463397,0.352181,2.08064},{-0.408895,0.280529,2.04919},{-0.147771,0.17349,2.17673},{-0.100429,0.391397,1.87494},{-0.125154,0.735275,1.97362},{-0.076999,0.833148,1.83602},{-0.31014,0.169258,2.17079},{-0.319744,0.384523,1.86409},{-0.313362,0.734925,1.95627},{-0.344045,0.828273,1.82804},{-0.221365,-0.40601,2.1583},{-0.159357,-0.425139,2.02239},{-0.145486,-0.458999,2.05998},{-0.138936,-0.455417,2.17857},{-0.188823,-0.460705,2.03717},{-0.280763,-0.473882,2.11751},{-0.233158,0.176178,2.1798},{-0.229384,0.00722323,2.19077},{-0.226015,-0.127698,2.18004},{-0.222712,-0.333658,2.1736},{-0.191358,-0.299481,2.1773},{-0.0616837,-0.263714,2.19516},{-0.0320325,-0.00483526,2.22539},{-0.0295941,0.200511,2.14601},{0.019159,0.275206,2.16601},{-0.0216646,0.330557,2.08991},{-0.024852,0.295904,2.05497},{-0.255155,-0.299835,2.17402},{-0.376716,-0.2715,2.17201},{-0.425043,-0.0092676,2.1904},{-0.428321,0.200767,2.11568},{-0.47454,0.280073,2.15117},{-0.460702,0.351005,2.08087},{-0.411754,0.293246,2.04165},{-0.146641,0.178213,2.18138},{-0.0984112,0.390059,1.8696},{-0.116921,0.737016,1.97533},{-0.0752415,0.83133,1.83044},{-0.311173,0.174343,2.17838},{-0.319111,0.379891,1.85933},{-0.312389,0.733574,1.95708},{-0.345933,0.827549,1.82741},{-0.220835,-0.409693,2.1585},{-0.148371,-0.430172,2.02586},{-0.135592,-0.462874,2.06574},{-0.137073,-0.455837,2.18586},{-0.177757,-0.466937,2.03988},{-0.275485,-0.481812,2.11535}};
    test_adaptive_constrained_skeleton_filter<BarPointFilter>(expected);
}
*/
TEST(AdaptiveConstrainedSkeletonFilterBasicSanityCheckRosePointFilter, BasicAssertions)
{
    std::vector<Point<double>> expected = {{-0.23315,0.171265,2.17361},{-0.229523,0.00518617,2.19278},{-0.225886,-0.128381,2.1885},{-0.223162,-0.331447,2.17561},{-0.192146,-0.29794,2.1807},{-0.0636616,-0.260943,2.18943},{-0.0327383,-0.00795043,2.23557},{-0.0387984,0.197247,2.16403},{0.0173699,0.267136,2.1537},{-0.0253986,0.321155,2.07947},{-0.0703645,0.288735,2.08823},{-0.255099,-0.297973,2.17674},{-0.373681,-0.266332,2.16527},{-0.429734,-0.0101757,2.19518},{-0.426251,0.198687,2.12569},{-0.462696,0.284368,2.15352},{-0.463397,0.352181,2.08064},{-0.408895,0.280529,2.04919},{-0.147771,0.17349,2.17673},{-0.100429,0.391397,1.87494},{-0.125154,0.735275,1.97362},{-0.076999,0.833148,1.83602},{-0.31014,0.169258,2.17079},{-0.319744,0.384523,1.86409},{-0.313362,0.734925,1.95627},{-0.344045,0.828273,1.82804},{-0.221365,-0.40601,2.1583},{-0.159357,-0.425139,2.02239},{-0.145486,-0.458999,2.05998},{-0.138936,-0.455417,2.17857},{-0.188823,-0.460705,2.03717},{-0.280763,-0.473882,2.11751},{-0.233158,0.176178,2.1798},{-0.229384,0.00722341,2.19077},{-0.226015,-0.127697,2.18003},{-0.222712,-0.333659,2.1736},{-0.191358,-0.299482,2.1773},{-0.0616837,-0.263714,2.19516},{-0.0320325,-0.00483526,2.22539},{-0.0295941,0.200511,2.14601},{0.019161,0.275208,2.16601},{-0.0216614,0.330561,2.08991},{-0.0248865,0.295911,2.05499},{-0.255155,-0.299835,2.17402},{-0.376716,-0.2715,2.17201},{-0.425043,-0.0092676,2.1904},{-0.428321,0.200767,2.11568},{-0.474558,0.280073,2.15117},{-0.460698,0.351004,2.08087},{-0.411756,0.293256,2.04165},{-0.146641,0.178213,2.18138},{-0.0984112,0.390059,1.8696},{-0.116921,0.737016,1.97533},{-0.0752415,0.83133,1.83044},{-0.311173,0.174343,2.17838},{-0.319111,0.379891,1.85933},{-0.312389,0.733574,1.95708},{-0.345933,0.827549,1.82741},{-0.220835,-0.409694,2.1585},{-0.148362,-0.430181,2.02586},{-0.135587,-0.462878,2.06574},{-0.137073,-0.455837,2.18587},{-0.177751,-0.466945,2.03988},{-0.275484,-0.481816,2.11535}};
    test_adaptive_constrained_skeleton_filter<RosePointFilter>(expected);
}
