// Contains Information about a specific skeleton configuration e.g. for the
// Azure Kinect
#include <vector>

struct SkeletonConfiguration {
    int joint_count;
};

struct ConstrainedSkeletonConfiguration : SkeletonConfiguration {
    std::vector<std::vector<int>> constrained_joint_groups;
};
