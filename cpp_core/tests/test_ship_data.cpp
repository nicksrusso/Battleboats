// Stage 2 test: exercise the CSV loaders against the real config files and
// assert known-good values. Plain <cassert> for now; we'll swap in a proper
// framework later.
#include "battleboats/shipyard/ship_data.h"

#include <cassert>
#include <iostream>

int main() {
    using namespace battleboats;

    // Absolute paths for now — fragile, cleaned up when the engine owns config.
    const std::string config = "/home/nick/Desktop/repos/Battleboats/battleboats/core/config/";
    const StatsTable stats = load_base_stats(config + "shipTypes.csv");
    const AttackModTable mods = load_attack_modifiers(config + "attackModifiers.csv");

    // --- base stats: values land in the right fields ---
    assert(stats.at(ShipType::Carrier).cost == 250);
    assert(stats.at(ShipType::Carrier).speed == 6);
    assert(stats.at(ShipType::Battleship).strength == 125.0);

    // --- the two CSV oddities parsed despite lowercase / trailing space ---
    assert(stats.count(ShipType::Merchant) == 1);  // "merchant" in the CSV
    assert(stats.count(ShipType::Landing) == 1);   // "Landing " in the CSV

    // --- attack modifiers: present pair, and the default-to-1.0 path ---
    assert(attack_modifier(mods, ShipType::Submarine, ShipType::Carrier) == 2.0);
    assert(attack_modifier(mods, ShipType::Carrier, ShipType::Battleship) == 1.5);
    assert(attack_modifier(mods, ShipType::Merchant, ShipType::Carrier) == 1.0);  // absent -> default

    std::cout << "ship_data tests passed\n";
    return 0;
}
