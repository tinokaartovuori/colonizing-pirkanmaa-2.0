/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: resourcemaps.h, contains different resource maps             #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/



#ifndef RESOURCEMAPS_H
#define RESOURCEMAPS_H

#include "basicresources.h"

namespace Course {

namespace ConstResourceMaps {

const ResourceMap EMPTY = {};

const ResourceMap NO_RESOURCES = {
    {BasicResource::MONEY, 0},
    {BasicResource::WOOD, 0},
    {BasicResource::METAL, 0},
    {BasicResource::STONE, 0}
};

const ResourceMap RESOURCE_LIMITS = {
    {BasicResource::MONEY, 9999999},
    {BasicResource::WOOD, 9999999},
    {BasicResource::STONE, 9999999},
    {BasicResource::METAL, 9999999}
};

const int UNIT_LIMITS = 999;


const ResourceMap STARTING_RESOURCES = {
    {BasicResource::MONEY, 400},
    {BasicResource::WOOD, 200},
    {BasicResource::STONE, 100},
    {BasicResource::METAL, 25}
};

// Tile - Forest
const ResourceMap FOREST_PRODUCTION = {
    {BasicResource::WOOD, 100},
    {BasicResource::STONE, 10}
};

const ResourceMap FOREST_CAPACITY= {
    {BasicResource::WOOD, 600},
    {BasicResource::STONE, 60}
};

// Tile - Abundant Forest
const ResourceMap ABUNDANT_FOREST_PRODUCTION ={
    {BasicResource::MONEY, 15},
};

const int FOREST_GROW_TIME = 5;

// Building - Farm
const ResourceMap FARM_BUILD_COST = {
    {BasicResource::MONEY, -100},
    {BasicResource::WOOD, -100},
    {BasicResource::METAL, -5}

};
const ResourceMap FARM_PRODUCTION = {
    {BasicResource::MONEY, 175}
};

const int FARM_GROW_TIME = 4;

// Building - Mine
const ResourceMap MINE_BUILD_COST = {
    {BasicResource::MONEY, -250},
    {BasicResource::WOOD, -250},
    {BasicResource::STONE, 200}
};
const ResourceMap MINE_PRODUCTION = {
    {BasicResource::MONEY, 20},
    {BasicResource::STONE, 30},
    {BasicResource::METAL, 20}
};


// Building - Hydroeletric Power Plant
const ResourceMap HEPP_BUILD_COST = {
    {BasicResource::MONEY, -360},
    {BasicResource::WOOD, -150},
    {BasicResource::STONE, -200},
    {BasicResource::METAL, -100}
};

const ResourceMap HEPP_PRODUCTION = {
    {BasicResource::MONEY, 40}
};

// Building - Nuclear Power Plant
const ResourceMap NUCLEARPP_BUILD_COST = {
    {BasicResource::MONEY, -1200},
    {BasicResource::WOOD, -200},
    {BasicResource::STONE, -500},
    {BasicResource::METAL, -500}
};

const ResourceMap NUCLEARPP_PRODUCTION = {
    {BasicResource::MONEY, 100}
};

// Building - Outpost
const ResourceMap OUTPOST_BUILD_COST = {
    {BasicResource::MONEY, -650},
    {BasicResource::WOOD, -300},
    {BasicResource::STONE, -300},
    {BasicResource::METAL, -300}
};

const ResourceMap OUTPOST_PRODUCTION = {
    {BasicResource::MONEY, -50},
    {BasicResource::METAL, -15}
};

const int OUTPOST_SOLDIER_VALUE = 3;

// Building - Bridge
const ResourceMap BRIDGE_BUILD_COST = {
    {BasicResource::MONEY, -100},
    {BasicResource::WOOD, -300},
    {BasicResource::STONE, -150}
};

const ResourceMap BRIDGE_PRODUCTION = {
    {BasicResource::WOOD, -5}
};

// Building - Neighborhood
const ResourceMap VILLAGE_BUILD_COST = {
    {BasicResource::MONEY, -200},
    {BasicResource::WOOD, -200},
    {BasicResource::STONE, -100},
    {BasicResource::METAL, -25}
};

const ResourceMap VILLAGE_PRODUCTION = {
    {BasicResource::MONEY, -10},
    {BasicResource::WOOD, -10},
    {BasicResource::STONE, -10}
};


const int VILLAGE_UNIT_VALUE = 3;

// Building - Mikontalo
const int MIKONTALO_UNIT_VALUE = 2;

// Building - HQ
const int HQ_UNIT_VALUE = 3;
const int HQ_SOLDIER_VALUE = 1;


// Worker
const ResourceMap BASIC_WORKER_COST = {
    {BasicResource::MONEY, -50}
};

const ResourceMap BASIC_WORKER_SALARY = {
    {BasicResource::MONEY, -5}
};

// Expert
const ResourceMap EXPERT_COST = {
    {BasicResource::MONEY, -250}
};
const ResourceMap EXPERT_SALARY = {
    {BasicResource::MONEY, -25}
};

// Soldier
const ResourceMap SOLDIER_COST = {
    {BasicResource::MONEY, -200},
    {BasicResource::METAL, -50}

};
const ResourceMap SOLDIER_SALARY = {
    {BasicResource::MONEY, -30}

};


}
}
#endif // RESOURCEMAPS_H
