/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: mikontalo.cpp, see mikotalo.h for the class's description          #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "mikontalo.h"

namespace Student {

Mikontalo::Mikontalo(const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
           const std::weak_ptr<Course::iObjectManager>& objectmanager,
           const std::shared_ptr<Course::PlayerBase> &owner,
           const Course::ResourceMap& buildcost,
           const Course::ResourceMap& production
           ):
    Course::BuildingBase(
        eventhandler,
        objectmanager,
        owner,
        buildcost,
        production,
        ConstDescriptionMaps::MIKONTALO_DESCRIPTION
        )
{
}

std::string Mikontalo::getType() const
{
    return "Mikontalo";
}


std::string Mikontalo::getExtraDescription() {
    return "<u>Effects:</u><br>+" +
            std::to_string(Course::ConstResourceMaps::MIKONTALO_UNIT_VALUE) +
            " Max Units";
}



} //namespace Student
