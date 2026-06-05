/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: farm.cpp, see farm.h for the class's description                   #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "farm.h"


namespace Course {

Farm::Farm(const std::weak_ptr<iGameEventHandler>& eventhandler,
           const std::weak_ptr<iObjectManager>& objectmanager,
           const std::weak_ptr<PlayerBase>& owner,
           const ResourceMap& buildcost,
           const ResourceMap& production
           ):
    BuildingBase(
        eventhandler, objectmanager,
        owner, buildcost, production, Student::ConstDescriptionMaps::FARM_DESCRIPTION),
        growthPhase_(1)
{
}

std::string Farm::getType() const
{
    return "Farm";
}

int Farm::getGrowthPhase()
{
    return growthPhase_;
}

void Farm::setGrowthPhase(int phase)
{
    growthPhase_ = phase;
    if (growthPhase_>=5) {
        growthPhase_ = 1;
    }
}

void Farm::resetFarm()
{
    setGrowthPhase(1);
    lockEventHandler()->updateAnimatedTileToStatic
                         (parentTile_.lock(), 1); //Frame is set to 1 graphically
}


} // namespace Course
