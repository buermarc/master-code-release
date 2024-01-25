#include <algorithm>
#include <filter/Point.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tclap/CmdLine.h>
#include <tuple>
#include <vector>

Point<double> azure_kinect_origin_lab_coords(Point<double> l_ak, Point<double> r_ak, Point<double> b_ak)
{
    Point<double> middle_between_left_and_right = l_ak + (r_ak - l_ak) / 2;

    auto result = (r_ak - l_ak).cross_product(b_ak - l_ak);
    return result;
}

template <typename T>
void print_vec(std::vector<T> vector)
{
    std::for_each(vector.cbegin(), vector.cend() - 1, [](auto ele) { std::cout << ele << ", " << std::endl; });
    std::cout << vector.back() << std::endl;
}
template <typename T>
void print_vec(std::string name, std::vector<T> vector)
{
    std::cout << "@name: ";
    std::for_each(vector.cbegin(), vector.cend() - 1, [](auto ele) { std::cout << ele << ", " << std::endl; });
    std::cout << vector.back() << std::endl;
}

std::string trimString(std::string str)
{
    const std::string whiteSpaces = " \t\n\r\f\v";
    // Remove leading whitespace
    size_t first_non_space = str.find_first_not_of(whiteSpaces);
    str.erase(0, first_non_space);
    // Remove trailing whitespace
    size_t last_non_space = str.find_last_not_of(whiteSpaces);
    str.erase(last_non_space + 1);
    return str;
}

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string> result;
    std::string line;
    std::getline(str, line);

    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, '\t')) {
        result.push_back(trimString(cell));
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty()) {
        // If there was a trailing comma then add an empty element.
        result.push_back("");
    }
    return result;
}

void read_marker_file(std::string file)
{

    std::ifstream csv_file(file);

    // Go through headers
    std::string key = "";
    do {
        auto header = getNextLineAndSplitIntoTokens(csv_file);
        if (header.size() > 0) {
            key = header.at(0);
        }
        for (auto element : header) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
        ;
    } while (key != "Frame");

    std::vector<double> timestamps;
    std::vector<Point<double>> l_ak;
    std::vector<Point<double>> r_ak;
    std::vector<Point<double>> b_ak;

    std::vector<Point<double>> l_sae;

    std::vector<Point<double>> l_hle;
    std::vector<Point<double>> l_usp;

    std::vector<Point<double>> r_hle;
    std::vector<Point<double>> r_usp;

    while (!csv_file.eof()) {
        auto results = getNextLineAndSplitIntoTokens(csv_file);
        if (results.size() == 1) {
            for (auto e : results) {
                std::cout << e << std::endl;
            }
            break;
        }
        timestamps.push_back(std::stod(results.at(1)));
        int i;

        i = 2;
        l_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));

        i += 3;
        r_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));

        i += 3;
        b_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));

        i += 3;
        l_sae.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));

        i += 3;
        l_hle.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));

        i += 3;
        l_usp.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));

        i += 3;
        r_hle.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));

        i += 3;
        r_usp.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));
    }

    print_vec("timestamps", timestamps);
    print_vec("l_ak", l_ak);
    print_vec("r_ak", r_ak);
    print_vec("b_ak", b_ak);
    print_vec("l_hle", l_hle);
    print_vec("l_usp", l_usp);
    print_vec("r_hle", r_hle);
    print_vec("r_usp", r_usp);
}

std::tuple<
    std::vector<double>,
    std::vector<Point<double>>,
    std::vector<Point<double>>,
    std::vector<Point<double>>>
read_force_plate_file(std::string force_plate_file)
{
    std::ifstream csv_file(force_plate_file);

    // Go through headers for file
    std::string key = "";
    do {
        auto header = getNextLineAndSplitIntoTokens(csv_file);
        if (header.size() > 0) {
            key = header.at(0);
        }
        for (auto element : header) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
        ;
    } while (key != "SAMPLE");

    std::vector<double> timestamps;
    std::vector<Point<double>> force;
    std::vector<Point<double>> moment;
    std::vector<Point<double>> cop;

    while (!csv_file.eof()) {
        auto results = getNextLineAndSplitIntoTokens(csv_file);
        if (results.size() == 1) {
            for (auto e : results) {
                std::cout << e << std::endl;
            }
            break;
        }
        timestamps.push_back(std::stod(results.at(1)));
        int i;

        i = 2;
        force.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));

        i += 3;
        moment.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));

        i += 3;
        cop.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))));
    }
    return std::make_tuple(timestamps, force, moment, cop);
}

void read_force_plate_files(std::string force_plate_file_f1, std::string force_plate_file_f2)
{
    auto [timestamp_f1, force_f1, moment_f1, com_f1] = read_force_plate_file(force_plate_file_f1);
    auto [timestamp_f2, force_f2, moment_f2, com_f2] = read_force_plate_file(force_plate_file_f2);

    print_vec("timestamp_f1", timestamp_f1);
    print_vec("force_f1", force_f1);
    print_vec("moment_f1", moment_f1);
    print_vec("com_f1", com_f1);
    print_vec("timestamp_f2", timestamp_f2);
    print_vec("force_f2", force_f2);
    print_vec("moment_f2", moment_f2);
    print_vec("com_f2", com_f2);
}

int main(int argc, char** argv)
{

    TCLAP::CmdLine cmd("Read tsv file from Qualisys.");

    TCLAP::ValueArg<std::string> tsv_file("i", "infile",
        "TSV File from qualisys", false, "",
        "string");

    cmd.add(tsv_file);
    cmd.parse(argc, argv);
    auto file = tsv_file.getValue();

    file.replace(file.find(".tsv"), sizeof(".tsv") - 1, "");
    std::stringstream marker_file_name;
    std::stringstream force_plate_file_name_f1;
    std::stringstream force_plate_file_name_f2;

    marker_file_name << file << ".tsv";
    force_plate_file_name_f1 << file << "_f_1.tsv";
    force_plate_file_name_f2 << file << "_f_2.tsv";

    read_marker_file(marker_file_name.str());
    read_force_plate_files(force_plate_file_name_f1.str(), force_plate_file_name_f2.str());

    std::cout << marker_file_name.str() << std::endl;
    std::cout << force_plate_file_name_f1.str() << std::endl;
    std::cout << force_plate_file_name_f2.str() << std::endl;
}
