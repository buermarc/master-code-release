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
    vvp m_predictions;
    bool m_enabled;

    std::string m_filter_type_name = "Unset";
    double m_measurement_error_factor = -1.0;

public:

    SkeletonSaver(bool enabled = true)
        : m_enabled(enabled)
    {
    }

    SkeletonSaver(vV timestamps, vp unfiltered_positions, vp filtered_positions, vp filtered_velocities, vp predictions)
        : m_timestamps(timestamps)
        , m_unfiltered_positions(unfiltered_positions)
        , m_filtered_positions(filtered_positions)
        , m_filtered_velocities(filtered_velocities)
        , m_predictions(predictions)
    {
    }

    void save_step(Value timestamp, vp unfiltered_positions, vp filtered_positions, vp filtered_velocities, vp predictions = vp())
    {
        m_timestamps.push_back(timestamp);
        m_unfiltered_positions.push_back(unfiltered_positions);
        m_filtered_positions.push_back(filtered_positions);
        m_filtered_velocities.push_back(filtered_velocities);
        m_predictions.push_back(predictions);
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

    vvp get_predictions()
    {
        return m_predictions;
    }

    void set_filter_type(std::string name )
    {
        this->m_filter_type_name = name;
    }

    void set_measurement_error_factor(double factor)
    {
        this->m_measurement_error_factor = factor;
    }

    double get_measurement_error_factor()
    {
        return this->m_measurement_error_factor;
    }

    json to_json() const
    {
        json _json;
        _json["timestamps"] = m_timestamps;
        _json["unfiltered_positions"] = m_unfiltered_positions;
        _json["filtered_positions"] = m_filtered_positions;
        _json["filtered_velocities"] = m_filtered_velocities;
        _json["predictions"] = m_predictions;
        _json["filter_type"] = m_filter_type_name;
        _json["measurement_error_factor"] = m_measurement_error_factor;
        return _json;
    }

    static SkeletonSaver<Value> from_json(json _json)
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
        ar& m_filter_type_name;
        ar& m_measurement_error_factor;
    }
};

template <typename Value>
void to_json(json& _json, const SkeletonSaver<Value>& saver)
{
    _json = saver.to_json();
}
