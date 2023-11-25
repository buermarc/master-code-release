#pragma once
#include <iostream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename Value>
class Point {
public:
    Value x;
    Value y;
    Value z;

    Point()
    {
        x = 0;
        y = 0;
        z = 0;
    }
    Point(Value m_x, Value m_y, Value m_z)
        : x(m_x)
        , y(m_y)
        , z(m_z)
    {
    }
    Point(Value* m_data)
        : x(m_data[0])
        , y(m_data[1])
        , z(m_data[2])
    {
    }

    template <typename U>
    friend std::ostream& operator<<(std::ostream& out, const Point<U>& point);

    template <typename U>
    friend bool operator==(const Point<U>& lhs, const Point<U>& rhs);

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& x;
        ar& y;
        ar& z;
    }
};

template <typename Value>
void to_json(json& _json, const Point<Value>& value)
{
    _json = json { std::vector { value.x, value.y, value.z } };
}

template <typename Value>
void from_json(const json& _json, Point<Value>& value)
{
    value.x = _json[0];
    value.y = _json[1];
    value.z = _json[2];
}

template <typename Value>
std::ostream& operator<<(std::ostream& out, const Point<Value>& point)
{
    // Since operator<< is a friend of the Point class, we can access Point's
    // members directly.
    out << "Point(" << point.x << ", " << point.y << ", " << point.z
        << ')'; // actual output done here

    return out;
}

template <typename Value>
bool operator==(const Point<Value>& lhs, const Point<Value>& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}
