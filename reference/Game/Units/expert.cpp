/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: expert.cpp, see expert.h for the class's description               #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/


#include "expert.h"

namespace Student {

Expert::Expert(const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
               const std::weak_ptr<Course::iObjectManager>& objectmanager,
               const std::weak_ptr<GameSettingsManager>& gamesettingsmanager,
               const std::weak_ptr<Course::PlayerBase>& owner,
               const std::weak_ptr<Course::TileBase>& tile):
    Course::UnitBase(
        eventhandler,
        objectmanager,
        gamesettingsmanager,
        owner,
        tile)
{
}

Expert::Expert(const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
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


std::string Expert::getType() const
{
    return "Expert";
}


Course::ResourceMap Expert::getSalary()
{
    return Course::ConstResourceMaps::EXPERT_SALARY;
}


Course::ResourceMap Expert::getCost()
{
    return Course::ConstResourceMaps::EXPERT_COST;
}


}

