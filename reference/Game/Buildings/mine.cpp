/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: mine.cpp, see mine.h for the class's description                   #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "mine.h"

namespace Student {

Mine::Mine(const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
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
        Student::ConstDescriptionMaps::MINE_DESCRIPTION
        )
{
}

std::string Mine::getType() const
{
    return "Mine";
}


} //namespace Student
