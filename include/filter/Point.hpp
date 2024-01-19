#pragma once
#include <iostream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <nlohmann/json.hpp>
#include <Eigen/Dense>

using Eigen::MatrixXd;


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

    const Point<Value> project_onto_plane(Point<Value>& point, Point<Value>& normed_n) {
        Point<Value> projected_point;
        Value sum = (point.x - this->x) * normed_n.x +
            (point.y - this->y) * normed_n.y +
            (point.z - this->z) * normed_n.z;
        std::cout << "sum: " << sum << std::endl;
        projected_point.x = this->x + sum * normed_n.x;
        projected_point.y = this->y + sum * normed_n.y;
        projected_point.z = this->z + sum * normed_n.z;
        return projected_point;
    }

    const Point<Value> cross_product(Point<Value>& other) {
        Point<Value> cross_product;
        cross_product.x = this->y*other.z - this->z*other.y;
        cross_product.y = this->z*other.x - this->x*other.z;
        cross_product.z = this->x*other.y - this->y*other.x;
        return cross_product;
    }

    const Value norm() {
        return std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
    }

    const Point<Value> normalized() {
        auto norm = this->norm();
        Point<double> copy(*this);
        return copy / norm;
    }

    Point<Value> operator+(Point<Value> const& other)
    {
        Point<Value> result;
        result.x = this->x + other.x;
        result.y = this->y + other.y;
        result.z = this->z + other.z;
        return result;
    }

    Point<Value> operator-(Point<Value> const& other)
    {
        Point<Value> result;
        result.x = this->x - other.x;
        result.y = this->y - other.y;
        result.z = this->z - other.z;
        return result;
    }

    Point<Value> operator*(Point<Value> const& other)
    {
        Point<Value> result;
        result.x = this->x * other.x;
        result.y = this->y * other.y;
        result.z = this->z * other.z;
        return result;
    }

    Point<Value> mat_mul(MatrixXd const& other)
    {
        assert(other.rows() == 3);
        assert(other.cols() == 3);
        Point<Value> result;
        result.x = this->x * other(0, 0) + this->y * other(0, 1) + this->z * other(0, 2);
        result.y = this->x * other(1, 0) + this->y * other(1, 1) + this->z * other(1, 2);
        result.z = this->x * other(2, 0) + this->y * other(2, 1) + this->z * other(2, 2);
        return result;
    }


    Point<Value> operator/(Point<Value> const& other)
    {
        Point<Value> result;
        result.x = this->x / other.x;
        result.y = this->y / other.y;
        result.z = this->z / other.z;
        return result;
    }

    Point<Value> operator+(Value const& value)
    {
        Point<Value> result;
        result.x = this->x + value;
        result.y = this->y + value;
        result.z = this->z + value;
        return result;
    }

    Point<Value> operator-(Value const& value)
    {
        Point<Value> result;
        result.x = this->x - value;
        result.y = this->y - value;
        result.z = this->z - value;
        return result;
    }

    Point<Value> operator*(Value const& value)
    {
        Point<Value> result;
        result.x = this->x * value;
        result.y = this->y * value;
        result.z = this->z * value;
        return result;
    }

    Point<Value> operator/(Value const& value)
    {
        Point<Value> result;
        result.x = this->x / value;
        result.y = this->y / value;
        result.z = this->z / value;
        return result;
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
