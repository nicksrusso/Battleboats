#pragma once
#include "battleboats/shipyard/ship_type.h"
#include "battleboats/shipyard/ship_stats.h"
#include "battleboats/map/pos.h"

namespace battleboats
{
    struct Ship
    {
        int id;
        ShipType type;
        ShipStats stats;
        int owner;
        Pos position;
        int tiles_moved_this_turn = 0;
        bool has_attacked = false;
        int cargo = 0;

        void reset_turn_flags()
        {
            tiles_moved_this_turn = 0;
            has_attacked = false;
        }
    };

}