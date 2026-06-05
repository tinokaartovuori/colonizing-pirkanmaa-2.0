/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: outpost.cpp, see outpost.h for the class's description             #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "outpost.h"
#include "Interfaces/iobjectmanager.h"
#include "Tiles/tilebase.h"

namespace Course {

Outpost::Outpost(
        const std::weak_ptr<iGameEventHandler>& eventhandler,
        const std::weak_ptr<iObjectManager>& objectmanager,
        const std::weak_ptr<PlayerBase>& owner,
        const ResourceMap& buildcost,
        const ResourceMap& production
        ):
    BuildingBase(eventhandler,
                 objectmanager,
                 owner,
                 buildcost,
                 production,
                 Student::ConstDescriptionMaps::OUTPOST_DESCRIPTION)
{
}

std::string Outpost::getType() const
{
    return "Outpost";
}

std::string Outpost::getExtraDescription() {
    return "<u>Effects:</u><br>+" + std::to_string(Course::ConstResourceMaps::OUTPOST_SOLDIER_VALUE) + " Max Soldiers";
}


} // namespace Course
