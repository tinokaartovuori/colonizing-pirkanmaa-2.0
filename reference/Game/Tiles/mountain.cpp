/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: mountain.cpp, see mountain.h for the class's description           #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "mountain.h"


namespace Student {

Mountain::Mountain(const Course::Coordinate& location,
                 int size_x,
                 int size_y,
                 const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
                 const std::weak_ptr<Course::iObjectManager>& objectmanager,
                 const unsigned int& max_units,
                 const Course::ResourceMap& production):
    TileBase(location,
             size_x,
             size_y,
             eventhandler,
             objectmanager,
             max_units,
             production,
             ConstDescriptionMaps::MOUNTAIN_DESCRIPTION)
{
}


std::string Mountain::getType() const
{
    return "Mountain";
}


std::vector<std::string> Mountain::getBuildableBuildings()
{
    return  {"Mine"};
}


void Mountain::generateResources()
{

    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Mine") {
            bool hasExpert = false;
            for (auto unit : getUnits()) {
                if (unit->getType() == "Expert") {
                    hasExpert = true;
                }
            }
            for (auto unit : getUnits()) {
                if (unit->getType() == "BasicWorker") {
                    owner_.lock()->addOrRemoveResources(getBuilding()
                                                        ->getProduction());
                    if (hasExpert) {
                        owner_.lock()->addOrRemoveResources
                                                (getBuilding()->getProduction());
                    }
                }
            }
        }
    }
}


Course::ResourceMap Mountain::getCurrentRevenue()
{
    Course::ResourceMap production = Course::ConstResourceMaps::NO_RESOURCES;
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Mine") {
            bool hasExpert = false;
            for (auto unit : getUnits()) {
                if (unit->getType() == "Expert") {
                    hasExpert = true;
                }
            }
            for (auto unit : getUnits()) {
                if (unit->getType() == "BasicWorker") {
                    production = mergeResourceMaps
                                 (production, getBuilding()->getProduction());
                    if (hasExpert) {
                        production = mergeResourceMaps
                                 (production, getBuilding()->getProduction());
                    }
                }
            }
        }
    }

    return production;
}


std::string Mountain::getExtraDescription()
{
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Mine") {
            for (auto unit : getUnits()) {
                if (unit->getType() == "Expert") {
                    return "Expert doubles the production rate.";
                }
            }
        }
    }

    return "";
}


} // namespace Student
