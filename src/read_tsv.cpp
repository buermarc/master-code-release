#include<string>
#include<vector>
#include<fstream>
#include<iostream>
#include<sstream>
#include<tclap/CmdLine.h>
#include<filter/Point.hpp>

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

void read_marker_file(std::string file) {

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
        std::cout << std::endl;;
    } while (key != "TRAJECTORY_TYPES");



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
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        r_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        b_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        l_sae.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        l_hle.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        l_usp.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        r_hle.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        r_usp.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );
    }
}

void read_force_files(std::string force_file_f_1, std::string force_file_f_2) {

    std::ifstream csv_file_1(force_file_f_1);
    std::ifstream csv_file_2(force_file_f_2);

    // Go through headers
    std::string key = "";
    do {
        auto header = getNextLineAndSplitIntoTokens(csv_file_1);
        if (header.size() > 0) {
            key = header.at(0);
        }
        for (auto element : header) {
            std::cout << element << " ";
        }
        std::cout << std::endl;;
    } while (key != "TRAJECTORY_TYPES");



    std::vector<double> timestamps;
    std::vector<Point<double>> l_ak;
    std::vector<Point<double>> r_ak;
    std::vector<Point<double>> b_ak;

    std::vector<Point<double>> l_sae;

    std::vector<Point<double>> l_hle;
    std::vector<Point<double>> l_usp;

    std::vector<Point<double>> r_hle;
    std::vector<Point<double>> r_usp;

    while (!csv_file_1.eof()) {
        auto results = getNextLineAndSplitIntoTokens(csv_file_1);
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
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        r_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        b_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        l_sae.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        l_hle.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        l_usp.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        r_hle.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        r_usp.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );
    }
}

int main(int argc, char** argv) {

    TCLAP::CmdLine cmd("Read tsv file from Qualisys.");

    TCLAP::ValueArg<std::string> tsv_file("i", "infile",
        "TSV File from qualisys", false, "",
        "string");

    cmd.add(tsv_file);
    cmd.parse(argc, argv);
    auto file = tsv_file.getValue();

    file.replace(file.find(".tsv"), sizeof(".tsv") - 1, "");
    std::stringstream marker_file_name;
    std::stringstream force_file_name_f_1;
    std::stringstream force_file_name_f_2;

    marker_file_name << file <<  ".tsv";
    force_file_name_f_1 << file << "_f_1.tsv";
    force_file_name_f_2 << file << "_f_2.tsv";

    read_marker_file(marker_file_name.str());

    std::cout << marker_file_name.str() << std::endl;
    std::cout << force_file_name_f_1.str() << std::endl;
    std::cout << force_file_name_f_2.str() << std::endl;

}
