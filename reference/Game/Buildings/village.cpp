/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: village.cpp, see village.h for the class's description             #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "village.h"

namespace Student {

Village::Village(const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
           const std::weak_ptr<Course::iObjectManager>& objectmanager,
           const std::weak_ptr<Course::PlayerBase>& owner,
           const Course::ResourceMap& buildcost,
           const Course::ResourceMap& production
           ):
    Course::BuildingBase(
        eventhandler,
        objectmanager,
        owner,
        buildcost,
        production,
        ConstDescriptionMaps::VILLAGE_DESCRIPTION
        )
{
}

std::string Village::getType() const
{
    return "Village";
}

std::string Village::getExtraDescription() {
    return "<u>Effects:</u><br>+" + std::to_string(Course::ConstResourceMaps::VILLAGE_UNIT_VALUE) + " Max Units";
}



} //namespace Student
