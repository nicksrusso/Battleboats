#pragma once
#include <cstdint>
#include <cctype>
#include <string>
#include <stdexcept>

namespace battleboats
{
    enum class ShipType : std::int8_t
    {
        Carrier,
        Battleship,
        Cruiser,
        Destroyer,
        Submarine,
        Merchant,
        Landing,
        Builder
    };

    inline std::string to_string(ShipType t)
    {
        switch (t)
        {
        case ShipType::Carrier:
            return "Carrier";
        case ShipType::Battleship:
            return "Battleship";
        case ShipType::Cruiser:
            return "Cruiser";
        case ShipType::Destroyer:
            return "Destroyer";
        case ShipType::Submarine:
            return "Submarine";
        case ShipType::Merchant:
            return "Merchant";
        case ShipType::Landing:
            return "Landing";
        case ShipType::Builder:
            return "Builder";
            throw std::invalid_argument("to_string: unknown ShipType");
        }
    }

    inline ShipType from_string(std::string s)
    {
        const std::string ws = " \t\r\n";
        const std::size_t start = s.find_first_not_of(ws);
        if (start == std::string::npos)
        {
            throw std::invalid_argument("from_string: empty ship type name");
        }
        const std::size_t end = s.find_last_not_of(ws);
        s = s.substr(start, end - start + 1);
        for (std::size_t i = 0; i < s.size(); ++i)
        {
            const unsigned char c = static_cast<unsigned char>(s[i]);
            s[i] = (i == 0) ? static_cast<char>(std::toupper(c))
                            : static_cast<char>(std::tolower(c));
        }

        if (s == "Carrier")
            return ShipType::Carrier;
        if (s == "Battleship")
            return ShipType::Battleship;
        if (s == "Cruiser")
            return ShipType::Cruiser;
        if (s == "Destroyer")
            return ShipType::Destroyer;
        if (s == "Submarine")
            return ShipType::Submarine;
        if (s == "Merchant")
            return ShipType::Merchant;
        if (s == "Landing")
            return ShipType::Landing;
        if (s == "Builder")
            return ShipType::Builder;
        throw std::invalid_argument("from_string: unknown ship type name: " + s);
    }
}
