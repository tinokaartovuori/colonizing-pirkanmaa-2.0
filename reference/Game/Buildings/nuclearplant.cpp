/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: nuclearplant.cpp, see nuclearplant.h for the class's description   #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "nuclearplant.h"

namespace Student {

NuclearPlant::NuclearPlant(const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
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
        Student::ConstDescriptionMaps::NUCLEAR_DESCRIPTION
        )
{
}

std::string NuclearPlant::getType() const
{
    return "Nuclear Power Plant";
}



} // namespace Student
