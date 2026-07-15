#pragma once

#include <string>
#include <unordered_map>

#include "battleboats/shipyard/ship_type.h"
#include "battleboats/shipyard/ship_stats.h"
#include "battleboats/shipyard/ship.h"

namespace battleboats
{

    // Base stats per ship type, loaded from shipTypes.csv.
    using StatsTable = std::unordered_map<ShipType, ShipStats>;

    // Attacker -> Defender -> strength multiplier, loaded from attackModifiers.csv.
    // Only combat ships appear; missing pairs default to 1.0 (see attack_modifier).
    using AttackModTable = std::unordered_map<ShipType, std::unordered_map<ShipType, double>>;

    // Parse shipTypes.csv into a full 8-entry table. Throws std::runtime_error if
    // the file can't be opened or any ShipType is missing a row.
    StatsTable load_base_stats(const std::string &csv_path);

    // Parse attackModifiers.csv into the nested multiplier table.
    AttackModTable load_attack_modifiers(const std::string &csv_path);

    // Multiplier on attacker's strength vs this defender. Returns 1.0 when the
    // (attacker, defender) pair is absent from the table.
    double attack_modifier(const AttackModTable &table, ShipType attacker, ShipType defender);

} // namespace battleboats
