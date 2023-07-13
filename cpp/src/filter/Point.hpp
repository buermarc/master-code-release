#pragma once
#include <iostream>

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
