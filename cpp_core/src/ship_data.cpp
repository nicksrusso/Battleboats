#include "battleboats/shipyard/ship_data.h"

#include <fstream>   // std::ifstream
#include <sstream>   // std::stringstream (line splitting)
#include <stdexcept> // std::runtime_error
#include <vector>

namespace battleboats
{

    // Split `line` on `delim` into fields. C++ has no str.split, so this is the
    // workhorse both loaders lean on.
    static std::vector<std::string> split(const std::string &line, char delim)
    {
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, delim))
        {
            fields.push_back(field);
        }
        return fields;
    }

    StatsTable load_base_stats(const std::string &csv_path)
    {
        StatsTable table;
        std::ifstream file(csv_path);
        if (!file)
        {
            throw std::runtime_error("shits fucked in the csv");
        }
        std::string line;
        std::getline(file, line);
        std::unordered_map<std::string, std::size_t> col;
        {
            const std::vector<std::string> headers = split(line, ',');
            for (std::size_t i = 0; i < headers.size(); i++)
            {
                col[headers[i]] = i;
            }
        }
        while (std::getline(file, line))
        {
            if (line.empty())
                continue;
            const std::vector<std::string> f = split(line, ',');
            const ShipType type = from_string(f[col.at("Ship Type")]);
            ShipStats s;
            s.cost = std::stoi(f[col.at("Cost")]);
            s.strength = std::stod(f[col.at("Strength")]);
            s.attack_range = std::stoi(f[col.at("attack range")]);
            s.speed = std::stoi(f[col.at("speed")]);
            s.visibility = std::stod(f[col.at("visibility coef")]);
            s.scouting = std::stod(f[col.at("scouting coef")]);

            table[type] = s;
        }

        const ShipType all[] = {
            ShipType::Carrier,
            ShipType::Battleship,
            ShipType::Cruiser,
            ShipType::Destroyer,
            ShipType::Submarine,
            ShipType::Merchant,
            ShipType::Landing,
            ShipType::Builder,
        };
        for (ShipType t : all)
        {
            if (table.find(t) == table.end())
            {
                throw std::runtime_error("load_base_stats: missing row for " + to_string(t));
            }
        }
        return table;
    }

    AttackModTable load_attack_modifiers(const std::string &csv_path)
    {
        AttackModTable table;
        std::ifstream file(csv_path);
        if (!file)
        {
            if (!file)
            {
                throw std::runtime_error("load_attack_modifiers: cannot open " + csv_path);
            }
        }

        std::string line;
        std::getline(file, line);
        std::vector<ShipType> defenders;
        {
            const std::vector<std::string> headers = split(line, ',');
            for (std::size_t i = 1; i < headers.size(); ++i)
            {
                if (!headers[i].empty())
                {
                    defenders.push_back(from_string(headers[i]));
                }
            }
        }
        while (std::getline(file, line))
        {
            if (line.empty())
                continue;
            const std::vector<std::string> f = split(line, ',');
            if (f.empty() || f[0].empty())
                continue;
            const ShipType attacker = from_string(f[0]);
            for (std::size_t i = 1; i < f.size() && (i - 1) < defenders.size(); ++i)
            {
                if (!f[i].empty())
                {
                    table[attacker][defenders[i - 1]] = std::stod(f[i]);
                }
            }
        }
        return table;
    }

    double attack_modifier(const AttackModTable &table, ShipType attacker, ShipType defender)
    {
        const auto a = table.find(attacker);
        if (a == table.end())
            return 1.0;
        const auto d = a->second.find(defender);
        if (d == a->second.end())
            return 1.0;
        return d->second;
    }

} // namespace battleboats
