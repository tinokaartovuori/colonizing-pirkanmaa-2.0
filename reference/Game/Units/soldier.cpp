/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: soldier.cpp, see soldier.h for the class's description             #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "soldier.h"

namespace Student {

Soldier::Soldier(const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
           const std::weak_ptr<Course::iObjectManager>& objectmanager,
           const std::weak_ptr<GameSettingsManager>& gamesettingsmanager,
           const std::weak_ptr<Course::PlayerBase>& owner,
           const std::weak_ptr<Course::TileBase>& tile
 ):
    Course::UnitBase(
        eventhandler,
        objectmanager,
        gamesettingsmanager,
        owner,
        tile)
{
}

Soldier::Soldier(const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
                 const std::weak_ptr<Course::iObjectManager> &objectmanager,
                 const std::weak_ptr<GameSettingsManager> &gamesettingsmanager,
                 const std::weak_ptr<Course::PlayerBase> &owner):
    Course::UnitBase(
        eventhandler,
        objectmanager,
        gamesettingsmanager,
        owner)
{
}

std::string Soldier::getType() const
{
    return "Soldier";
}


Course::ResourceMap Soldier::getSalary()
{
    return Course::ConstResourceMaps::SOLDIER_SALARY;
}


Course::ResourceMap Soldier::getCost()
{
    return Course::ConstResourceMaps::SOLDIER_COST;
}


} //namespace Student


