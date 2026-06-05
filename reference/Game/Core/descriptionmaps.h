/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: descriptionmaps.h, contains description-strings              #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef DESCRIPTIONMAPS_H
#define DESCRIPTIONMAPS_H

#include "Core/resourcemaps.h"

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace Student {

namespace ConstDescriptionMaps {


const std::string FARM_DESCRIPTION =
        "This is a lovely place. Crops can be grown here. "
        "Never leave the crops alone or they will die! "
        "Crops can be harvested after <u>" +
        std::to_string(Course::ConstResourceMaps::FARM_GROW_TIME) +
        " rounds</u>.";

const std::string BRIDGE_DESCRIPTION =
        "You can use this to cross rivers. "
        "Bridge needs a little maintenance every round.";

const std::string VILLAGE_DESCRIPTION =
        "Increases the amount of units you can have by <u>" +
        std::to_string(Course::ConstResourceMaps::VILLAGE_UNIT_VALUE) +
        "</u>! For soldiers you will need something else...";

const std::string HEPP_DESCRIPTION =
        "Uses water flow to produce energy. This hydroelectric power plant is "
        "kinda advanced stuff so you will need at least one expert here.";

const std::string NUCLEAR_DESCRIPTION =
        "Nuclear power plant is the most efficent power plant available. "
        "This is very dangerous and advanced technology so at least one expert "
        "is required here.";

const std::string MINE_DESCRIPTION =
        "Mine is a very fun place to work. You can be in the dark whole day and"
        " mine some stone and metal. "
        "Expert can make your work much better though.";

const std::string OUTPOST_DESCRIPTION =
        "Good place for soldiers to hang out. Increses the amount of soldiers you"
        " can have by <u>" +
        std::to_string(Course::ConstResourceMaps::OUTPOST_SOLDIER_VALUE) +
        "</u>! Enemy cannot directly attack this building.";


const std::string HEADQUARTERS_DESCRIPTION =
        "This is the heart of your region. If enemy gets here you lose. "
        "Because of the ultimate security no units allowed here.";

const std::string BROKEN_HEADQUARTERS_DESCRIPTION =
        "Headquarters that got destroyed...";


/////////////////////////////

const std::string FOREST_DESCRIPTION =
        "Forest is the only way to get wood. "
        "More workers make chopping down forest faster. "
        "Forest will grow back but you can build on top of it after cutting.";

const std::string GRASSLAND_DESCRIPTION =
        "Just a bunch of grass and stuff... There is a lot of stuff that can"
        " be built here.";

const std::string MIKONTALO_DESCRIPTION =
        "Some big house that seems to be founded in 1978-1980. "
        "Owning this awesome place increases the amount of units you can have"
        " by <u>" +
        std::to_string(Course::ConstResourceMaps::MIKONTALO_UNIT_VALUE) +
        "</u>. No need for maintenance either.";

const std::string MOUNTAIN_DESCRIPTION =
        "I love high places. Maybe there is something valueable inside this big rock thingy... "
        "Consider building a mine here.";

const std::string RIVER_DESCRIPTION_1 =
        "I like how it flows... Maybe there is a way to get over it... "
        "Or maybe the flow can be used to generate power.";

const std::string RIVER_DESCRIPTION_2 =
        "Seems to be a bit too curvy place for a bridge... The flow is not that"
        " great either.";

const std::string ABUNDANT_FOREST_DESCRIPTION =
        "A lush forest where you can forage some juicy fruits.";

//////////////////////////

const std::string FARM_SHOP_DESCRIPTION =
        "Worker can grow crops. Crops can be harvested every <u>" +
        std::to_string(Course::ConstResourceMaps::FARM_GROW_TIME) +
        "</u> rounds. Never leave the crops alone!";

const std::string BRIDGE_SHOP_DESCRIPTION =
        "You can use this to cross some rivers! Remember to do the maintenace.";

const std::string VILLAGE_SHOP_DESCRIPTION =
        "Increases the amount of units you can have by <u>" +
        std::to_string(Course::ConstResourceMaps::VILLAGE_UNIT_VALUE) +
        "</u>! Soldiers will need something else though.";

const std::string HEPP_SHOP_DESCRIPTION =
        "Produces a lot of money! This power plant requires at least one expert.";

const std::string NUCLEAR_SHOP_DESCRIPTION =
        "Produces a ton of money! You will need at least one expert to do so.";

const std::string MINE_SHOP_DESCRIPTION =
        "You can mine some stone and metal! Expert can increase the efficency.";

const std::string OUTPOST_SHOP_DESCRIPTION =
        "Increases the amount of soldiers you can have by <u>" +
        std::to_string(Course::ConstResourceMaps::OUTPOST_SOLDIER_VALUE) +
        "</u>. Also protects tile directly next to it.";

}



}
#endif // DESCRIPTIONMAPS_H
