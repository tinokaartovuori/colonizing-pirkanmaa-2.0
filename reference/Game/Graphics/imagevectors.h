/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: freesceneitem.h, contains different vectors that that have   #
#                        file paths to image files                   #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef IMAGEVECTORS_H
#define IMAGEVECTORS_H

#include <vector>
#include <string>


namespace ImageVectors {

const std::vector<std::string> CLICKEDTILEBORDER = {":Images/selectionborder.png"};

const std::vector<std::string> MOUSEHOVERBORDER = {":Images/tilemousehover_1.png",
                                                   ":Images/tilemousehover_2.png"};
const std::vector<std::string> TILEOWNERBORDERS = {":Images/playeroneborder_n.png",
                                                ":Images/playertwoborder_n.png",
                                                ":Images/playerthreeborder_n.png",
                                                ":Images/playerfourborder_n.png"};
const std::vector<std::string> FOREST_1 = {":Images/forest_1_1.png",
                                         ":Images/forest_1_2.png",
                                         ":Images/forest_1_3.png"};
const std::vector<std::string> FOREST_2 = {":Images/forest_2_1.png",
                                         ":Images/forest_2_2.png",
                                         ":Images/forest_2_3.png"};
const std::vector<std::string> FOREST_STUMPS = {":Images/foreststumps.png",};

const std::vector<std::string> GRASSLAND = {":Images/grassland.png"};
const std::vector<std::string> MIKONTALO = {":Images/mikontalo.png"};
const std::vector<std::string> ABUNDANT_FOREST = {":Images/abundant_forest_1.png",
                                                 ":Images/abundant_forest_2.png",
                                                 ":Images/abundant_forest_3.png"};
const std::vector<std::string> MOUNTAIN = {":Images/mountain.png"};
const std::vector<std::string> MOUNTAIN_FOREST = {":Images/mountain_f_1.png",
                                           ":Images/mountain_f_2.png",
                                           ":Images/mountain_f_3.png"};
const std::vector<std::string> HEADQUARTERSONE = {":Images/headquarters1_3.png",
                                           ":Images/headquartersplayerone2.png",
                                           ":Images/headquarters1_3.png",
                                           ":Images/headquartersplayerone4.png"};
const std::vector<std::string> HEADQUARTERSTWO = {":Images/headquarters1_3.png",
                                           ":Images/headquartersplayertwo2.png",
                                           ":Images/headquarters1_3.png",
                                           ":Images/headquartersplayertwo4.png"};
const std::vector<std::string> HEADQUARTERSTHREE = {":Images/headquarters1_3.png",
                                           ":Images/headquartersplayerthree2.png",
                                           ":Images/headquarters1_3.png",
                                           ":Images/headquartersplayerthree4.png"};
const std::vector<std::string> HEADQUARTERSFOUR = {":Images/headquarters1_3.png",
                                           ":Images/headquartersplayerfour2.png",
                                           ":Images/headquarters1_3.png",
                                           ":Images/headquartersplayerfour4.png"};
const std::vector<std::string> HEADQUARTERSDESTROYED =
                                            {":Images/headquartersDestroyed.png"};

const std::vector<std::string> OUTPOST = {":Images/outpost_1.png",
                                           ":Images/outpost_2.png",
                                           ":Images/outpost_3.png"};
const std::vector<std::string> HYDROPOWERNS = {":Images/hydropower1NS.png",
                                               ":Images/hydropower2NS.png"};
const std::vector<std::string> HYDROPOWERWE = {":Images/hydropower1WE.png",
                                               ":Images/hydropower2WE.png"};
const std::vector<std::string> NUCLEARPLANT = {":Images/nuclearPlant1.png",
                                               ":Images/nuclearPlant2.png"};
const std::vector<std::string> VILLAGE = {":Images/village.png"};
const std::vector<std::string> BRIDGENS = {":Images/bridgeNS.png"};
const std::vector<std::string> BRIDGEWE = {":Images/bridgeWE.png"};
const std::vector<std::string> MINE = {":Images/mine.png"};
const std::vector<std::string> FARM = {":Images/farm1.png",
                                       ":Images/farm2.png",
                                       ":Images/farm3.png",
                                       ":Images/farm4.png"};


const std::vector<std::string> BASICWORKER = {":Images/basicworker_1.png",
                                             ":Images/basicworker_2.png"};
const std::vector<std::string> EXPERT = {":Images/expert_1.png",
                                             ":Images/expert_2.png"};
const std::vector<std::string> SOLDIER = {":Images/soldier_1.png",
                                             ":Images/soldier_2.png"};

const std::vector<std::string> BASICWORKER_SWIM = {":Images/basicworker_swim_1.png",
                                             ":Images/basicworker_swim_2.png"};
const std::vector<std::string> EXPERT_SWIM = {":Images/expert_swim_1.png",
                                             ":Images/expert_swim_2.png"};
const std::vector<std::string> SOLDIER_SWIM = {":Images/soldier_swim_1.png",
                                             ":Images/soldier_swim_2.png"};

const std::vector<std::string> COVER_BORDER = {":Images/tile_cover_border.png"};

const std::vector<std::string> MENU = {":Images/menu_bg.png"};
const std::vector<std::string> CONTAINER = {":Images/container_2_2.png"};
const std::vector<std::string> BUTTON = {":Images/button_1_2.png"};

const std::vector<std::string> RED = {":Images/red.png"};
const std::vector<std::string> BLUE = {":Images/blue.png"};
const std::vector<std::string> PURPLE = {":Images/purple.png"};
const std::vector<std::string> YELLOW = {":Images/yellow.png"};

const std::vector<std::string> BAR_RED = {":Images/color_bar_red.png"};
const std::vector<std::string> BAR_BLUE = {":Images/color_bar_blue.png"};
const std::vector<std::string> BAR_PURPLE = {":Images/color_bar_purple.png"};
const std::vector<std::string> BAR_YELLOW = {":Images/color_bar_yellow.png"};
const std::vector<std::string> BAR_NEUTRAL = {":Images/color_bar_neutral.png"};

const std::vector<std::string> BLOCKED_TILE = {":Images/blocked_tile.png"};

const std::vector<std::string> MONEY = {":Images/money.png"};
const std::vector<std::string> WOOD = {":Images/wood.png"};
const std::vector<std::string> STONE = {":Images/stone.png"};
const std::vector<std::string> METAL = {":Images/metal.png"};

const std::vector<std::string> MULTI = {":Images/multi_0.png",
                                        ":Images/multi_1.png",
                                        ":Images/multi_2.png",
                                        ":Images/multi_3.png",
                                        ":Images/multi_4.png",
                                        ":Images/multi_5.png",
                                        ":Images/multi_6.png",
                                        ":Images/multi_7.png",
                                        ":Images/multi_8.png",
                                       };

//Vector name shows the direction in compass points W=west, N=north etc...
const std::vector<std::string> RIVER_EW = {":Images/river_ew_1.png",
                                           ":Images/river_ew_2.png"};
const std::vector<std::string> RIVER_NS = {":Images/river_ns_1.png",
                                           ":Images/river_ns_2.png"};
const std::vector<std::string> RIVER_NW = {":Images/river_nw_1.png",
                                           ":Images/river_nw_2.png"};
const std::vector<std::string> RIVER_NE = {":Images/river_ne_1.png",
                                           ":Images/river_ne_2.png"};
const std::vector<std::string> RIVER_SW = {":Images/river_sw_1.png",
                                           ":Images/river_sw_2.png"};
const std::vector<std::string> RIVER_SE = {":Images/river_se_1.png",
                                           ":Images/river_se_2.png"};

}

#endif // ITEMIMAGEVECTORS_H

