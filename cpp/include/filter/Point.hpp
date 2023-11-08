#pragma once
#include <iostream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

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
