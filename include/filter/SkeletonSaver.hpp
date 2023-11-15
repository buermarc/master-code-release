#pragma once
#include <algorithm>
#include <filter/com.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using json = nlohmann::json;

template <typename Value>
class SkeletonSaver {

    typedef std::vector<std::vector<Point<Value>>> vvp;
    typedef std::vector<Point<Value>> vp;
    typedef std::vector<Value> vV;

    vV m_timestamps;
    vvp m_unfiltered_positions;
    vvp m_filtered_positions;
    vvp m_filtered_velocities;
    bool m_enabled;

public:
    SkeletonSaver(bool enabled = true)
        : m_enabled(enabled)
    {
    }

    SkeletonSaver(vV timestamps, vp unfiltered_positions, vp filtered_positions, vp filtered_velocities)
        : m_timestamps(timestamps)
        , m_unfiltered_positions(unfiltered_positions)
        , m_filtered_positions(filtered_positions)
        , m_filtered_velocities(filtered_velocities)
    {
    }

    void save_step(Value timestamp, vp unfiltered_positions, vp filtered_positions, vp filtered_velocities)
    {
        m_timestamps.push_back(timestamp);
        m_unfiltered_positions.push_back(unfiltered_positions);
        m_filtered_positions.push_back(filtered_positions);
        m_filtered_velocities.push_back(filtered_velocities);
    }

    bool saver_enabled()
    {
        return m_enabled;
    }

    void enable_saver()
    {
        m_enabled = true;
    }

    void disable_saver()
    {
        m_enabled = true;
    }

    vV get_timestamps()
    {
        return m_timestamps;
    }

    vvp get_unfiltered_positions()
    {
        return m_unfiltered_positions;
    }

    vvp get_filtered_positions()
    {
        return m_filtered_positions;
    }

    vvp get_filtered_velocities()
    {
        return m_filtered_velocities;
    }

    json to_json()
    {
        json _json;
        _json["timestamps"] = m_timestamps;
        _json["unfiltered_positions"] = m_unfiltered_positions;
        _json["filtered_positions"] = m_filtered_positions;
        _json["filtered_velocities"] = m_filtered_velocities;
        return _json;
    }

    json from_json(json _json)
    {
        return SkeletonSaver<Value>(_json["timestamps"], _json["unfiltered_positions"], _json["filtered_positions"], _json["filtered_velocities"]);
    }

    json read_from_json_file(std::string path)
    {
        std::ifstream infile(path);
        json _json;
        infile >> _json;
        return _json;
    }

    void write_to_json_file(std::string path)
    {
        std::ofstream outfile(path);
        outfile << to_json();
    }

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& m_timestamps;
        ar& m_unfiltered_positions;
        ar& m_filtered_positions;
        ar& m_filtered_velocities;
    }
};
